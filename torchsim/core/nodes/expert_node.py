import logging
import operator
from typing import List, Optional, Dict, Tuple, Any

import torch
from functools import reduce
from torchsim.core.graph.hierarchical_observable_node import HierarchicalObservableNode
from torchsim.core.graph.inverse_pass_packet import InversePassInputPacket
from torchsim.core.graph.invertible_node import InversePassOutputPacket
from torchsim.core.graph.node_base import NodeValidationException
from torchsim.core.graph.slot_container import Inputs
from torchsim.core.graph.slot_container_base import GenericMemoryBlocks
from torchsim.core.graph.slots import InputSlot
from torchsim.core.graph.unit import InvertibleUnit
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import NUMBER_OF_CONTEXT_TYPES
from torchsim.core.models.expert_params_props import SpatialPoolerParamsProps, ExpertParamsProps, TemporalPoolerParamsProps
from torchsim.core.models.flock.expert_flock import ExpertFlock
from torchsim.core.models.spatial_pooler import ExpertParams, SPFlockLearning
from torchsim.core.models.temporal_pooler import TPFlockLearning, TPFlockForwardAndBackward
from torchsim.core.nodes.flock_node_utils import create_sp_learn_observables, create_tp_forward_observables, \
    create_tp_learn_observables
from torchsim.core.nodes.internals.learning_switchable import LearningSwitchable
from torchsim.core.nodes.spatial_pooler_node import SPFlockInputsSection, SPFlockReconstructionSection, \
    SPFlockInternalsSection, OutputShapeProvider, SPFlockForwardClustersSlotSection
from torchsim.core.nodes.temporal_pooler_node import TPFlockInputsContextAndRewardSection, TPFlockOutputsSection, \
    TPFlockInputsDataSection, TPFlockInternalsSection
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver
from torchsim.core.utils.inverse_projection_utils import get_inverse_projections_for_all_clusters, \
    replace_cluster_ids_with_projections
from torchsim.core.utils.tensor_utils import view_dim_as_dims
from torchsim.gui.observer_system import Observable
from torchsim.gui.observers.cluster_observer import ClusterObserverExpertFlock
from torchsim.gui.observers.hierarchical_observer import HierarchicalObserver
from torchsim.gui.observers.memory_block_observer import CustomTensorObserver
from torchsim.utils.node_utils import derive_flock_shape, TensorShapePatternMatcher
from torchsim.utils.seed_utils import set_global_seeds

logger = logging.getLogger(__name__)


class ExpertFlockUnit(InvertibleUnit):

    def __init__(self, creator, params: ExpertParams):
        super().__init__(creator.device)

        self.flock = self._create_flock(params, creator)

    @staticmethod
    def _create_flock(params, creator):
        return ExpertFlock(params, creator)

    def step(self, data: torch.Tensor, context: torch.Tensor, reward: torch.Tensor):
        self.flock.run(data, context, reward)

    def inverse_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        """Computes the inverse projection using TP and SP.
        """

        # tp_output = self.flock.tp_flock.inverse_projection(tensor)
        # return self.flock.sp_flock.inverse_projection(tp_output)
        return self.flock.sp_flock.inverse_projection(tensor)

    def set_sp_learning(self, value: bool):
        self.flock.sp_flock.enable_learning = value

    def set_tp_learning(self, value: bool):
        self.flock.tp_flock.enable_learning = value

    def _save(self, saver: Saver):
        saver.description['sp_enable_learning'] = self.flock.sp_flock.enable_learning
        saver.description['tp_enable_learning'] = self.flock.tp_flock.enable_learning

    def _load(self, loader: Loader):
        self.flock.sp_flock.enable_learning = loader.description['sp_enable_learning']
        self.flock.tp_flock.enable_learning = loader.description['tp_enable_learning']


class ExpertFlockInputs(Inputs):
    """Class which holds the inputs for an ExpertFlockNode.

    Args:
        owner (ExpertFlockNode): The node to which these inputs belong to.
    """

    def __init__(self, owner: 'ExpertFlockNode'):
        super().__init__(owner)
        self.sp = SPFlockInputsSection(self)
        self.tp = TPFlockInputsContextAndRewardSection(self)


class ExpertFlockOutputs(GenericMemoryBlocks['ExpertFlockNode']):
    """Class which holds the outputs for an ExpertFlockNode.

    Args:
        owner (ExpertFlockNode): The node to which these outputs belong to.
    """

    def __init__(self, owner: 'ExpertFlockNode'):
        super().__init__(owner)
        self.sp = SPFlockReconstructionSection(self)
        self.tp = TPFlockOutputsSection(self)
        self.output_context = self.create("Expert_context_outputs")

    def prepare_slots(self, unit: ExpertFlockUnit):
        data_input_shape = self._owner.inputs.sp.data_input.tensor.shape
        flock_size = self._owner.params.flock_size
        self.sp.prepare_slots_from_flock(unit.flock.sp_flock, self._owner, data_input_shape, flock_size)
        self.tp.prepare_slots_from_flock(unit.flock.tp_flock, data_input_shape, flock_size)
        self.output_context.tensor = unit.flock.output_context

        flock_shape = derive_flock_shape(data_input_shape, flock_size)

        # self.current_reconstructed_input.reshape_tensor(shape=flock_shape, dim=0)

        self.output_context.reshape_tensor(shape=flock_shape, dim=0)


class ExpertSPFlockInternalsSection(SPFlockInternalsSection, SPFlockForwardClustersSlotSection):
    pass


class ExpertTPFlockInternalsSection(TPFlockInternalsSection, TPFlockInputsDataSection):
    pass


class ExpertFlockInternals(GenericMemoryBlocks['ExpertFlockNode']):

    def __init__(self, owner: 'ExpertFlockNode'):
        super().__init__(owner)
        self.sp = ExpertSPFlockInternalsSection(self)
        self.tp = ExpertTPFlockInternalsSection(self)

    def prepare_slots(self, unit: ExpertFlockUnit):
        data_input_shape = self._owner.inputs.sp.data_input.tensor.shape
        flock_size = self._owner.params.flock_size
        self.sp.prepare_slots_from_flock(unit.flock.sp_flock, self._owner, data_input_shape, flock_size)
        self.tp.prepare_slots_from_flock(unit.flock.tp_flock, data_input_shape, flock_size)

        # SP/TP boundary. Just pass the reference through
        self.tp.data_input.tensor = self.sp.forward_clusters.tensor


class ExpertFlockNode(HierarchicalObservableNode[ExpertFlockInputs, ExpertFlockInternals, ExpertFlockOutputs],
                      LearningSwitchable, OutputShapeProvider):
    """Node which represents a flock of TA Experts with a 1-1 correspondence of Spatial poolers to Temporal poolers."""

    _unit: ExpertFlockUnit

    inputs: ExpertFlockInputs

    memory_blocks: ExpertFlockInternals
    outputs: ExpertFlockOutputs
    _seed: int
    _sp_params_props: SpatialPoolerParamsProps
    _tp_params_props: TemporalPoolerParamsProps
    _expert_params_props: ExpertParamsProps

    @property
    def projected_values(self) -> Optional[torch.Tensor]:
        if self._unit is None:
            return None

        return view_dim_as_dims(self._unit.flock.sp_flock.cluster_centers, self.get_receptive_field_shape())

    @property
    def projection_input(self) -> InputSlot:
        return self.inputs.sp.data_input

    def __init__(self, params: ExpertParams, seed: int = None, name="ExpertFlock"):
        """Initializes the node.

        Args:
            params (ExpertParams): The parameters with which to initialize the flock.
            seed (int, optional): An integer seed, defaults to None.
            name (str, optional): The name of the node, defaults to 'ExpertFlock'.
        """
        inputs, memory_blocks, outputs = self.get_memory_blocks()
        HierarchicalObservableNode.__init__(self, name=name, inputs=inputs, memory_blocks=memory_blocks,
                                            outputs=outputs)

        self.params = params.clone()
        self._seed = seed
        self._create_params_props()

    def get_memory_blocks(self):
        return ExpertFlockInputs(self), ExpertFlockInternals(self), ExpertFlockOutputs(self)

    def _create_unit(self, creator: TensorCreator) -> ExpertFlockUnit:
        self._derive_params()
        set_global_seeds(self._seed)
        return ExpertFlockUnit(creator, self.params)

    def validate(self):
        """Checks if all inputs have valid shapes.

        Rewards have to be either one float which will be translated into a tuple (positive, negative) reward and used
        for all experts, or separate tuples one for each expert in either (flock_size, 2) or (flock_shape, 2).
        Context input should be either
        (flock_size, n_providers, NUMBER_OF_CONTEXT_TYPES, self.params.temporal.incoming_context_size) or
        (flock_shape, n_providers, NUMBER_OF_CONTEXT_TYPES, self.params.temporal.incoming_context_size).
        """

        flock_shape = derive_flock_shape(self.inputs.sp.data_input.tensor.shape,
                                         self.params.flock_size)

        reward_input = self.inputs.tp.reward_input.tensor
        if reward_input is not None:
            expected_shapes = [(1,), (self.params.flock_size, 2), flock_shape + (2,), (2,)]
            if reward_input.size() not in expected_shapes:
                raise NodeValidationException(f"Reward input has unexpected shape {reward_input.size()}, "
                                              f"expected one of {expected_shapes}.")

        # Validate context_input shape
        context_input = self.inputs.tp.context_input.tensor
        if context_input is not None:
            matcher_pattern = (
                TensorShapePatternMatcher.Sum(self.params.flock_size),
                TensorShapePatternMatcher.Sum(self.params.temporal.n_providers, greedy=True),
                TensorShapePatternMatcher.Exact((NUMBER_OF_CONTEXT_TYPES,)),
                TensorShapePatternMatcher.Sum(self.params.temporal.incoming_context_size, greedy=True)
            )
            matcher = TensorShapePatternMatcher(matcher_pattern)
            if not matcher.matches(context_input.shape):
                pattern_str = ", ".join(map(str, matcher_pattern))
                raise NodeValidationException(
                    f"Context input has unexpected shape {list(context_input.shape)}, "
                    f"expected pattern: [{pattern_str}]")

    def _derive_params(self):
        """Derive the params of the node from the input shape."""
        data_input = self.inputs.sp.data_input.tensor
        self.params.spatial.input_size = data_input.numel() // self.params.flock_size
        if self.inputs.tp.context_input.tensor is not None:
            context_tensor_shape = self.inputs.tp.context_input.tensor.shape
            # context_input is of shape: (*flock_sizes, *providers, CONTEXT_TYPES, context_size)
            matcher_pattern = (
                TensorShapePatternMatcher.Sum(self.params.flock_size),
                TensorShapePatternMatcher.TrailingAny(),
            )
            matcher = TensorShapePatternMatcher(matcher_pattern)
            matches = matcher.matches(context_tensor_shape)
            assert matches is True  # The correct format is checked in validate
            # Group 0 is flock_sizes, now compute product of n_providers and extract context size
            dims = matcher.groups[1]
            self.params.temporal.n_providers = reduce(operator.mul, dims[:-2], 1)  # context_tensor_shape[-3]
            self.params.temporal.incoming_context_size = dims[-1]  # context_tensor_shape[-1]


    def get_receptive_field_shape(self) -> Tuple[Any]:
        flock_shape = derive_flock_shape(self.inputs.sp.data_input.tensor.shape,
                                         self.params.flock_size)
        n_flock_shape_dims = len(flock_shape)

        receptive_field_shape = self.inputs.sp.data_input.tensor.size()[n_flock_shape_dims:]

        return receptive_field_shape

    @property
    def output_shape(self) -> Tuple[Any]:
        return self.get_receptive_field_shape()

    def _step(self):
        if self.inputs.tp.reward_input.tensor is None:
            input_rewards = None
        else:
            input_rewards = self.inputs.tp.reward_input.tensor

        context_view = None if self.inputs.tp.context_input.tensor is None else \
            self.inputs.tp.context_input.tensor.view(self.params.flock_size, self.params.temporal.n_providers, NUMBER_OF_CONTEXT_TYPES, -1)
        self._unit.step(self.inputs.sp.data_input.tensor.view(self.params.flock_size, -1),
                        context_view,
                        input_rewards)

    def _on_initialization_change(self):
        self._create_params_props()

    def _create_params_props(self):
        self._sp_params_props = SpatialPoolerParamsProps(self.params.spatial,
                                                         self._unit.flock.sp_flock if self._unit is not None else None)
        self._tp_params_props = TemporalPoolerParamsProps(self.params.temporal,
                                                          self._unit.flock.tp_flock if self._unit is not None else None)
        self._expert_params_props = ExpertParamsProps(self.params, self._unit)

    def get_properties(self):
        properties = super().get_properties()
        return properties + [
            *self._expert_params_props.get_properties(),
            # self._prop_builder.collapsible_header("SP", True),
            *self._sp_params_props.get_properties(),
            *self._tp_params_props.get_properties()
        ]

    def switch_learning(self, on: bool):
        self.params.spatial.enable_learning = on
        self.params.temporal.enable_learning = on
        self._unit.set_sp_learning(on)
        self._unit.set_tp_learning(on)

    def _inverse_projection(self, data: InversePassOutputPacket) -> List[InversePassInputPacket]:
        if data.slot == self.outputs.tp.projection_outputs:
            # Only calculate for the expert output
            # projected = self._unit.inverse_projection(data.tensor)
            projected = self._unit.inverse_projection(data.tensor.view(
                self._unit.flock.sp_flock.forward_clusters.shape))
            # Change the last dimension into the original input dimensions.
            projected = projected.view(self.inputs.sp.data_input.tensor.shape)

            return [InversePassInputPacket(projected, self.inputs.sp.data_input)]

        return []

    def _get_process_sp_learn(self) -> Optional[SPFlockLearning]:
        return None if self._unit is None else self._unit.flock.sp_flock.learn_process

    def _get_process_tp_trained_forward(self) -> Optional[TPFlockForwardAndBackward]:
        return None if self._unit is None else self._unit.flock.tp_flock.trained_forward_process

    def _get_process_tp_learn(self) -> Optional[TPFlockLearning]:
        return None if self._unit is None else self._unit.flock.tp_flock.learn_process

    def _get_observables(self) -> Dict[str, Observable]:
        result = super()._get_observables()

        for i in range(self.params.flock_size):
            result[f'Custom.Hierarchical.expert_{i}'] = HierarchicalObserver(self, i)
            result[f'Custom.Cluster.expert_{i}'] = self._create_cluster_observer(i)

        result['Custom.SP_passive_predicted_reconstructed_input'] = CustomTensorObserver(
            self._single_step_scoped_cache.create(self._tensor_provider_passive_predicted_reconstructed_input)
        )

        result['Custom.TP_projection_output_reconstruction_aggregated'] = CustomTensorObserver(
            self._single_step_scoped_cache.create(self._tensor_provider_tp_output_projection_reconstruction_aggregated)
        )
        result['Custom.SP_frequent_seqs_reconstruction'] = CustomTensorObserver(
            self._single_step_scoped_cache.create(self._tensor_provider_sp_frequent_seqs_reconstruction)
        )
        result['Custom.TP_projection_output_reconstruction'] = CustomTensorObserver(
            self._single_step_scoped_cache.create(self._tensor_provider_tp_output_projection_reconstruction)
        )

        flock_size = self.params.flock_size
        sp_learn_flock_size = self._get_sp_learn_flock_size()

        result.update(create_sp_learn_observables(sp_learn_flock_size, self._get_process_sp_learn))
        result.update(create_tp_forward_observables(flock_size, self._get_process_tp_trained_forward))
        result.update(create_tp_learn_observables(flock_size, self._get_process_tp_learn))

        return result

    def _create_cluster_observer(self, number):
        return ClusterObserverExpertFlock(self, number)

    def _get_sp_learn_flock_size(self):
        """What is the real flock size when learning (flock_size or 1 in conv SP)."""

        return self.params.flock_size

    def _save(self, saver: Saver):
        super()._save(saver)

        saver.description['sp_enable_learning'] = self.params.spatial.enable_learning
        saver.description['tp_enable_learning'] = self.params.temporal.enable_learning

    def _load(self, loader: Loader):
        super()._load(loader)

        self.params.spatial.enable_learning = loader.description['sp_enable_learning']
        self.params.temporal.enable_learning = loader.description['tp_enable_learning']

    def _tensor_provider_passive_predicted_reconstructed_input(self) -> Optional[torch.Tensor]:
        if self._unit is None:
            return None

        next_seq_position = self._unit.flock.tp_flock.seq_lookbehind
        clusters = self._unit.flock.tp_flock.passive_predicted_clusters_outputs[:, next_seq_position, :]
        result = self._unit.flock.sp_flock.inverse_projection(clusters)
        result = result.view(self.inputs.sp.data_input.tensor.shape)
        return result

    def _tensor_provider_tp_output_projection_reconstruction_aggregated(self) -> Optional[torch.Tensor]:
        if self._unit is None:
            return None

        projection_output = self._unit.flock.tp_flock.projection_outputs
        result = self._unit.flock.sp_flock.inverse_projection(projection_output)
        result = result.view(self.inputs.sp.data_input.tensor.shape)
        return result

    def _tensor_provider_tp_output_projection_reconstruction(self) -> Optional[torch.Tensor]:
        if self._unit is None:
            return None

        # fixed to the first expert and the first input projection
        projections = self._get_inverse_projections(expert_id=0, projection_id=0)
        projection_output = self._unit.flock.tp_flock.projection_outputs

        projection_dims = len(projections.shape[1:])
        view_dims = [*projection_output.shape, *([1]*projection_dims)]
        result = projections * projection_output.clone().view(view_dims)
        return result

    def _tensor_provider_sp_frequent_seqs_reconstruction(self) -> Optional[torch.Tensor]:
        if self._unit is None:
            return None

        # fixed to the first expert and the first input projection
        projections = self._get_inverse_projections(expert_id=0, projection_id=0)
        return replace_cluster_ids_with_projections(self.memory_blocks.tp.frequent_seqs.tensor, projections)

    def _get_inverse_projections(self, expert_id: int, projection_id: int) -> torch.Tensor:
        all_projections = get_inverse_projections_for_all_clusters(self, expert_id)
        return torch.stack(all_projections[projection_id])

