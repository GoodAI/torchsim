import logging
from abc import abstractmethod
from typing import List, Optional, Dict, Tuple, Any

import torch
from torchsim.core.graph.hierarchical_observable_node import HierarchicalObservableNode
from torchsim.core.graph.inverse_pass_packet import InversePassInputPacket
from torchsim.core.graph.invertible_node import InversePassOutputPacket
from torchsim.core.graph.node_base import NodeBase
from torchsim.core.graph.slot_container import MemoryBlocks, Inputs, InputsSection, MemoryBlocksSection
from torchsim.core.graph.slot_container_base import GenericMemoryBlocks
from torchsim.core.graph.slots import InputSlot
from torchsim.core.graph.unit import InvertibleUnit
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params_props import SpatialPoolerParamsProps
from torchsim.core.models.spatial_pooler import SPFlock, ExpertParams, SPFlockLearning
from torchsim.core.nodes.flock_node_utils import create_sp_learn_observables
from torchsim.core.nodes.internals.learning_switchable import LearningSwitchable
from torchsim.core.utils.tensor_utils import view_dim_as_dims
from torchsim.gui.observables import Observable
from torchsim.gui.observers.cluster_observer import ClusterObserverSPFlock
from torchsim.gui.observers.flock_process_observable import FlockProcessObservable
from torchsim.gui.observers.hierarchical_observer import HierarchicalObserver
from torchsim.utils.node_utils import derive_flock_shape
from torchsim.utils.seed_utils import set_global_seeds

logger = logging.getLogger(__name__)


class OutputShapeProvider:
    @property
    @abstractmethod
    def output_shape(self) -> Tuple[Any]:
        pass


class SpatialPoolerFlockUnit(InvertibleUnit):

    def __init__(self, creator: TensorCreator, params: ExpertParams):
        super().__init__(creator.device)
        self._params = params
        self.flock = self._create_flock(params, creator)

    @staticmethod
    def _create_flock(params, creator):
        return SPFlock(params, creator)

    def step(self, data: torch.Tensor):
        indices = self.flock.forward_learn(data)
        if self._params.compute_reconstruction:
            self.flock.reconstruct(indices)

    def inverse_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.flock.inverse_projection(tensor)


class SPMemoryBlocksSection(MemoryBlocksSection):
    def prepare_slots_from_flock(self, flock: SPFlock, output_shape_provider: OutputShapeProvider,
                                 data_input_shape: Tuple[Any], flock_size: int):
        pass


class SPFlockInputsSection(InputsSection):
    """
    Attributes:
        data_input: Input to the flock is mandatory and it should be a tensor of the following dimensions:
        [X1, X2, ..., Xn, Y1, Y2, ..., Ym], where X1 * X2 * ... * Xn = flock_size, Y1, Y2, ..., Ym are dimensions of
        the data to each expert, n >= 1, and m >= 1. So for example, a grid of 2 times 2 experts, each looking at a
        MNIST image of size 28, 28, 1 would have an input with dimensions [2, 2, 28, 28, 1].
    """
    def __init__(self, container: Inputs):
        super().__init__(container)
        self.data_input = self.create("SP_data_input")


class SPFlockInputs(Inputs):
    """Class which holds the inputs for a SpatialPoolerFlockNode.

    Args:
        owner (SpatialPoolerFlockNode): The node to which these inputs belong to.
    """

    def __init__(self, owner: 'SpatialPoolerFlockNode'):
        super().__init__(owner)
        self.sp = SPFlockInputsSection(self)

    def prepare_slots(self, unit: SpatialPoolerFlockUnit):
        pass


class SPFlockReconstructionSection(SPMemoryBlocksSection):
    def __init__(self, container: GenericMemoryBlocks[NodeBase]):
        super().__init__(container)
        self.current_reconstructed_input = self.create("SP_current_reconstructed_input")
        self.predicted_reconstructed_input = self.create("SP_predicted_reconstructed_input")

    def prepare_slots_from_flock(self, flock: SPFlock, output_shape_provider: OutputShapeProvider,
                                 data_input_shape: Tuple[Any], flock_size: int):
        super().prepare_slots_from_flock(flock, output_shape_provider, data_input_shape, flock_size)
        self.current_reconstructed_input.tensor = flock.current_reconstructed_input
        self.predicted_reconstructed_input.tensor = flock.predicted_reconstructed_input

        # Reshape some of the tensors to match the input space.
        self.current_reconstructed_input.reshape_tensor(output_shape_provider.output_shape)
        self.predicted_reconstructed_input.reshape_tensor(output_shape_provider.output_shape)

        flock_shape = derive_flock_shape(data_input_shape, flock_size)

        self.current_reconstructed_input.reshape_tensor(shape=flock_shape, dim=0)
        self.predicted_reconstructed_input.reshape_tensor(shape=flock_shape, dim=0)


class SPFlockForwardClustersSlotSection(SPMemoryBlocksSection):
    def __init__(self, container: MemoryBlocks):
        super().__init__(container)
        self.forward_clusters = self.create("SP_forward_clusters")

    def prepare_slots_from_flock(self, flock: SPFlock, output_shape_provider: OutputShapeProvider,
                                 data_input_shape: Tuple[Any], flock_size: int):
        super().prepare_slots_from_flock(flock, output_shape_provider, data_input_shape, flock_size)
        self.forward_clusters.tensor = flock.forward_clusters
        flock_shape = derive_flock_shape(data_input_shape, flock_size)
        self.forward_clusters.reshape_tensor(shape=flock_shape, dim=0)


class SPFlockOutputsSection(SPFlockReconstructionSection, SPFlockForwardClustersSlotSection):
    pass


class SPFlockOutputs(GenericMemoryBlocks['SpatialPoolerFlockNode']):
    """Class which holds the outputs for a SpatialPoolerFlockNode.

    Args:
        owner (SpatialPoolerFlockNode): The node to which these outputs belong to.
    """

    def __init__(self, owner: 'SpatialPoolerFlockNode'):
        super().__init__(owner)
        self.sp = SPFlockOutputsSection(self)

    def prepare_slots(self, unit: SpatialPoolerFlockUnit):
        self.sp.prepare_slots_from_flock(unit.flock, self._owner, self._owner.inputs.sp.data_input.tensor.shape,
                                         self._owner.params.flock_size)


class SPFlockInternalsSection(SPMemoryBlocksSection):
    def __init__(self, container: GenericMemoryBlocks[NodeBase]):
        # SP class observables
        super().__init__(container)
        self.cluster_centers = self.create("SP_cluster_centers")
        self.cluster_boosting_durations = self.create("SP_cluster_boosting_durations")
        self.prev_boosted_clusters = self.create("SP_prev_boosted_clusters")
        self.cluster_center_targets = self.create("SP_cluster_center_targets")
        self.cluster_center_deltas = self.create("SP_cluster_center_deltas")
        self.boosting_targets = self.create("SP_boosting_targets")
        self.predicted_clusters = self.create("SP_predicted_clusters")
        self.execution_counter_forward = self.create("SP_execution_counter_forward")
        self.execution_counter_learning = self.create("SP_execution_counter_learning")

        # Buffer observables
        self.buffer_total_data_written = self.create("SP_buffer_total_data_written")
        self.buffer_inputs = self.create_buffer("SP_buffer_inputs")
        self.buffer_clusters = self.create_buffer("SP_buffer_clusters")

    def prepare_slots_from_flock(self, flock: SPFlock, output_shape_provider: OutputShapeProvider,
                                 data_input_shape: Tuple[Any], flock_size: int):
        super().prepare_slots_from_flock(flock, output_shape_provider, data_input_shape, flock_size)
        self.cluster_centers.tensor = flock.cluster_centers
        self.cluster_boosting_durations.tensor = flock.cluster_boosting_durations
        self.prev_boosted_clusters.tensor = flock.prev_boosted_clusters
        self.cluster_center_targets.tensor = flock.cluster_center_targets
        self.cluster_center_deltas.tensor = flock.cluster_center_deltas
        self.boosting_targets.tensor = flock.boosting_targets
        self.predicted_clusters.tensor = flock.predicted_clusters
        self.execution_counter_forward.tensor = flock.execution_counter_forward
        self.execution_counter_learning.tensor = flock.execution_counter_learning

        # Buffer observables
        self.buffer_total_data_written.tensor = flock.buffer.total_data_written
        self.buffer_inputs.buffer = flock.buffer.inputs
        self.buffer_clusters.buffer = flock.buffer.clusters

        # Reshape some of the tensors to match the input space.
        receptive_field_shape = output_shape_provider.output_shape

        # SP Observables
        self.cluster_centers.reshape_tensor(receptive_field_shape)
        self.cluster_center_targets.reshape_tensor(receptive_field_shape)
        self.cluster_center_deltas.reshape_tensor(receptive_field_shape)

        # SP Buffer
        self.buffer_inputs.reshape_tensor(receptive_field_shape)


class SPFlockInternals(GenericMemoryBlocks['SpatialPoolerFlockNode']):
    """Class which holds the internals for a SpatialPoolerFlockNode.

    Args:
        owner (SpatialPoolerFlockNode): The node to which these internals belong to.
    """

    def __init__(self, owner: 'SpatialPoolerFlockNode'):
        super().__init__(owner)
        self.sp = SPFlockInternalsSection(self)

    def prepare_slots(self, unit: SpatialPoolerFlockUnit):
        self.sp.prepare_slots_from_flock(unit.flock, self._owner, self._owner.inputs.sp.data_input.tensor.shape,
                                         self._owner.params.flock_size)


class SpatialPoolerFlockNode(HierarchicalObservableNode[SPFlockInputs, SPFlockInternals, SPFlockOutputs],
                             LearningSwitchable, OutputShapeProvider):
    """Node which represents a flock of TA Spatial poolers.

    Args:
        params (ExpertParams): The parameters with which to initialize the flock. Only the spatial part is used.
        seed (int, optional): An integer seed, defaults to None.
        name (str, optional): The name of the node, defaults to 'SPFlock'.
    """

    _unit: SpatialPoolerFlockUnit
    inputs: SPFlockInputs
    memory_blocks: SPFlockInternals
    outputs: SPFlockOutputs
    params: ExpertParams
    _seed: int
    _sp_params_props: SpatialPoolerParamsProps

    @property
    def projected_values(self) -> Optional[torch.Tensor]:
        if self._unit is None:
            return None

        return view_dim_as_dims(self._unit.flock.cluster_centers, self.get_receptive_field_shape())

    @property
    def projection_input(self) -> InputSlot:
        return self.inputs.sp.data_input

    def __init__(self, params: ExpertParams, seed: int = None, name="SPFlock"):
        inputs, memory_blocks, outputs = self.get_memory_blocks()
        HierarchicalObservableNode.__init__(self, name=name, inputs=inputs, memory_blocks=memory_blocks,
                                            outputs=outputs)

        self._seed = seed
        self.params = params.clone()
        self._create_params_props()

    def get_memory_blocks(self):
        return SPFlockInputs(self), SPFlockInternals(self), SPFlockOutputs(self)

    def _create_unit(self, creator: TensorCreator) -> SpatialPoolerFlockUnit:
        self._derive_params()
        set_global_seeds(self._seed)
        return SpatialPoolerFlockUnit(creator, self.params)

    # TODO (Refactor): DRY this (see expert_node).
    def _derive_params(self):
        """Derive the params of the node from the input shape."""
        data_input = self.inputs.sp.data_input.tensor
        self.params.spatial.input_size = data_input.numel() // self.params.flock_size

    # TODO (Refactor): DRY this (see expert_node).
    def get_receptive_field_shape(self):
        flock_shape = derive_flock_shape(self.inputs.sp.data_input.tensor.shape,
                                                                self.params.flock_size)
        n_flock_shape_dims = len(flock_shape)

        receptive_field_shape = self.inputs.sp.data_input.tensor.size()[n_flock_shape_dims:]

        return receptive_field_shape

    # TODO (Refactor): DRY this (see expert_node).
    def validate(self):
        """Checks if the input can be correctly parsed."""

        if self.params.spatial.cluster_boost_threshold < self.params.n_cluster_centers:
            logger.warning("Spatial pooler: number of cluster centers is greater than cluster boost threshold,"
                           " which means that unless batch_size is large, it is probable that even some useful clusters"
                           " will get boosted!")

        # validate flock shape - just to be sure, it is actually called even before in get_receptive_field_shape()
        derive_flock_shape(self.inputs.sp.data_input.tensor.shape, self.params.flock_size)

    @property
    def output_shape(self) -> Tuple[Any]:
        return self.get_receptive_field_shape()

    def _step(self):
        self._unit.step(self.inputs.sp.data_input.tensor.view(self.params.flock_size, -1))

    def _on_initialization_change(self):
        self._create_params_props()

    def _create_params_props(self):
        self._sp_params_props = SpatialPoolerParamsProps(self.params.spatial,
                                                         self._unit.flock if self._unit is not None else None)

    def get_properties(self):
        return self._sp_params_props.get_properties()

    def switch_learning(self, on):
        self._sp_params_props.enable_learning = on

    def _inverse_projection(self, data: InversePassOutputPacket) -> List[InversePassInputPacket]:
        if data.slot == self.outputs.sp.forward_clusters:
            # Only calculate for the expert output
            projected = self._unit.inverse_projection(data.tensor.view(self._unit.flock.forward_clusters.shape))
            # Change the last dimension into the original input dimensions.
            projected = projected.view(self.inputs.sp.data_input.tensor.shape)

            return [InversePassInputPacket(projected, self.inputs.sp.data_input)]

        return []

    def _get_process_learn(self):
        if self._unit is None:
            return None

        return self._unit.flock.learn_process

    def _create_learn_observable(self, getter):
        return FlockProcessObservable(self.params.flock_size,
                                      lambda: self._get_process_learn(),
                                      getter)

    def _get_process_sp_learn(self) -> Optional[SPFlockLearning]:
        return None if self._unit is None else self._unit.flock.learn_process

    def _get_observables(self) -> Dict[str, Observable]:
        result = super()._get_observables()

        sp_learn_flock_size = self._get_sp_learn_flock_size()

        result.update(create_sp_learn_observables(sp_learn_flock_size, self._get_process_sp_learn))

        for i in range(self.params.flock_size):
            result[f'Custom.Hierarchical.expert_{i}'] = HierarchicalObserver(self, i)
            result[f'Custom.Cluster.expert_{i}'] = self._create_cluster_observer(i)

        return result

    def _create_cluster_observer(self, number):
        return ClusterObserverSPFlock(self, number)

    # TODO: DRY (see ExpertNode)
    def _get_sp_learn_flock_size(self):
        """What is the real flock size when learning (flock_size or 1 in conv SP)."""

        return self.params.flock_size
