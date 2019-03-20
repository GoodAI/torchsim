from typing import Dict, Any, Tuple, Optional

import torch
from torchsim.core.graph.node_base import NodeBase

from torchsim.core.graph.slot_container import Inputs, MemoryBlocksSection, InputsSection
from torchsim.core.graph.slot_container_base import GenericMemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeWithInternalsBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params_props import TemporalPoolerParamsProps
from torchsim.core.models.temporal_pooler import TPFlock, ExpertParams, TPFlockForwardAndBackward, TPFlockLearning
from torchsim.core.nodes.flock_node_utils import create_tp_forward_observables, \
    create_tp_learn_observables
from torchsim.core.nodes.internals.learning_switchable import LearningSwitchable
from torchsim.gui.observables import Observable
from torchsim.utils.node_utils import derive_flock_shape
from torchsim.utils.seed_utils import set_global_seeds


class TemporalPoolerFlockUnit(Unit):

    def __init__(self, creator: TensorCreator, params: ExpertParams):
        super().__init__(creator.device)

        self.flock = TPFlock(params, creator)

    def step(self, input_clusters: torch.Tensor, context: torch.Tensor, rewards: torch.Tensor):
        self.flock.forward_learn(input_clusters, context, rewards)


class TPMemoryBlocksSection(MemoryBlocksSection):
    def prepare_slots_from_flock(self, flock: TPFlock, data_input_shape: Optional[Tuple[Any]], flock_size: int):
        pass


class TPFlockInputsDataSection(InputsSection):
    def __init__(self, container: Inputs):
        super().__init__(container)
        self.data_input = self.create("TP_data_input")


class TPFlockInputsContextAndRewardSection(InputsSection):
    """Context and reward inputs for the temporal pooler.

    Attributes:
        context_input: Optional, dimensions: [X1, X2, ..., Xn, 2, context_size]
        reward_input: Optional, dimensions: [1] or [X1, X2, ..., Xn, 2] or [X1 * X2 * ... * Xn, 2]

    X1 * X2 * ... * Xn = flock_size, Y1, Y2, ..., Ym are dimensions of the data to each expert, n >= 1, and m >= 1.
    """
    def __init__(self, container: Inputs):
        super().__init__(container)
        self.context_input = self.create("TP_context_input")
        self.reward_input = self.create("TP_reward_input")


class TPFlockInputsSection(TPFlockInputsDataSection, TPFlockInputsContextAndRewardSection):
    pass


class TPFlockInputs(Inputs):
    """Inputs to the TPFlockNode.

    `tp_data_input` needs to be connected, `tp_context_input` and `tp_reward_input` are optional and can be omitted.
    """

    def __init__(self, owner):
        super().__init__(owner)
        self.tp = TPFlockInputsSection(self)


class TPFlockOutputsSection(TPMemoryBlocksSection):
    def __init__(self, container: GenericMemoryBlocks[NodeBase]):
        super().__init__(container)
        self.projection_outputs = self.create("TP_projection_outputs")
        self.action_outputs = self.create("TP_action_outputs")
        self.best_matching_context = self.create("TP_best_matching_context")

    def prepare_slots_from_flock(self, flock: TPFlock, data_input_shape: Optional[Tuple[Any]], flock_size: int):
        super().prepare_slots_from_flock(flock, data_input_shape, flock_size)
        self.projection_outputs.tensor = flock.projection_outputs
        self.action_outputs.tensor = flock.action_rewards

        self.action_outputs.tensor = flock.action_outputs
        self.projection_outputs.tensor = flock.projection_outputs
        self.best_matching_context.tensor = flock.best_matching_context

        if (data_input_shape is not None) and (flock_size is not None):
            flock_shape = derive_flock_shape(data_input_shape, flock_size)
            self.projection_outputs.reshape_tensor(shape=flock_shape, dim=0)


class TPFlockOutputs(GenericMemoryBlocks['TemporalPoolerFlockNode']):
    """outputs of the TPFlockNode.

    `tp_projection_outputs` - representation of the input augmented by temporal information.
                              Typically send to the parents.
    `tp_action_outputs` - desired state to which the TPFlockNode would like to get in order to fulfill goals
                          received in the context and reward inputs.
    """

    def __init__(self, owner):
        super().__init__(owner)
        self.tp = TPFlockOutputsSection(self)

    def prepare_slots(self, unit: TemporalPoolerFlockUnit):
        self.tp.prepare_slots_from_flock(unit.flock, None, self._owner.params.flock_size)


class TPFlockInternalsSection(TPMemoryBlocksSection):
    def __init__(self, container: GenericMemoryBlocks[NodeBase]):
        super().__init__(container)
        self.all_encountered_seqs = self.create("TP_all_encountered_seqs")
        # ['flock_size', 'tp_max_encountered_seqs'] -> 'discounted_count'
        # Number of occurrences of all encountered sequences. Each learning step values are normalized so the sum of all
        # occurrences is at most tp_forgetting_limit
        self.all_encountered_seq_occurrences = self.create("TP_all_encountered_seq_occurrences")
        # ['flock_size', 'tp_n_frequent_seqs', 'tp_seq_length'] -> 'cluster_id'
        # Most frequent sequences used in forward/backward process, ordered by occurrence
        self.frequent_seqs = self.create("TP_frequent_seqs")
        # ['flock_size', 'tp_n_frequent_seqs'] -> 'discounted_count'
        # Number of occurrences of most frequent sequences. This is subset of all_encountered_seq_occurrences
        # - top tp_n_frequent_seqs are taken
        self.frequent_seq_occurrences = self.create("TP_frequent_seq_occurrences")
        # ['flock_size', 'tp_n_frequent_seqs', 'tp_seq_length', 'n_providers' (parents), 'tp_context_size'] -> 'probability'
        # Probability of individual symbols of frequent sequences to be seen in particular context when the particular
        # context is on (== 1).
        self.frequent_context_likelihoods = self.create("TP_frequent_context_likelihoods")
        # ['flock_size', 'tp_max_encountered_seqs', 'tp_seq_length', 'tp_context_size']
        self.all_encountered_context_occurrences = self.create("TP_all_encountered_context_occurrences")
        self.frequent_exploration_attempts = self.create("TP_frequent_exploration_attempts")
        self.frequent_exploration_results = self.create("TP_frequent_exploration_results")
        self.all_encountered_exploration_attempts = self.create("TP_all_encountered_exploration_attempts")
        self.all_encountered_exploration_success_rates = self.create("TP_all_encountered_exploration_success_rates")
        self.input_context = self.create("TP_input_context")
        self.passive_predicted_clusters = self.create("TP_passive_predicted_clusters")
        self.execution_counter_forward = self.create("TP_execution_counter_forward")
        self.execution_counter_learning = self.create("TP_execution_counter_learning")
        self.frequent_rewards_punishments = self.create("TP_frequent_rewards_punishments")
        self.frustration = self.create("TP_frustration")

        self.seq_likelihoods_by_context = self.create("TP_seq_likelihoods_by_context")

        # Buffer observables
        self.buffer_total_data_written = self.create("TP_buffer_total_data_written")
        self.buffer_outputs = self.create_buffer("TP_buffer_outputs")
        self.buffer_seq_probs = self.create_buffer("TP_buffer_seq_probs")
        self.buffer_clusters = self.create_buffer("TP_buffer_clusters")
        self.buffer_contexts = self.create_buffer("TP_buffer_contexts")
        self.buffer_actions = self.create_buffer("TP_buffer_actions")
        self.buffer_exploring = self.create_buffer("TP_buffer_exploring")
        self.buffer_rewards_punishments = self.create_buffer("TP_buffer_rewards_punishments")

    def prepare_slots_from_flock(self, flock: TPFlock, data_input_shape: Optional[Tuple[Any]], flock_size: int):
        super().prepare_slots_from_flock(flock, data_input_shape, flock_size)
        self.all_encountered_seqs.tensor = flock.all_encountered_seqs
        self.all_encountered_seq_occurrences.tensor = flock.all_encountered_seq_occurrences
        self.frequent_seqs.tensor = flock.frequent_seqs
        self.frequent_seq_occurrences.tensor = flock.frequent_seq_occurrences
        self.frequent_context_likelihoods.tensor = flock.frequent_context_likelihoods
        self.all_encountered_context_occurrences.tensor = flock.all_encountered_context_occurrences
        self.frequent_exploration_attempts.tensor = flock.frequent_exploration_attempts
        self.frequent_exploration_results.tensor = flock.frequent_exploration_results
        self.all_encountered_exploration_attempts.tensor = flock.all_encountered_exploration_attempts
        self.all_encountered_exploration_success_rates.tensor = flock.all_encountered_exploration_results
        self.input_context.tensor = flock.input_context
        self.passive_predicted_clusters.tensor = flock.passive_predicted_clusters_outputs
        self.execution_counter_forward.tensor = flock.execution_counter_forward
        self.execution_counter_learning.tensor = flock.execution_counter_learning
        self.seq_likelihoods_by_context.tensor = flock.seq_likelihoods_by_context
        self.frequent_rewards_punishments.tensor = flock.frequent_rewards_punishments
        self.frustration.tensor = flock.frustration

        # Buffer observables
        self.buffer_total_data_written.tensor = flock.buffer.total_data_written
        self.buffer_outputs.buffer = flock.buffer.outputs
        self.buffer_seq_probs.buffer = flock.buffer.seq_probs
        self.buffer_clusters.buffer = flock.buffer.clusters
        self.buffer_contexts.buffer = flock.buffer.contexts
        self.buffer_actions.buffer = flock.buffer.actions
        self.buffer_exploring.buffer = flock.buffer.exploring
        self.buffer_rewards_punishments.buffer = flock.buffer.rewards_punishments


class TPFlockInternals(GenericMemoryBlocks['TemporalPoolerFlockNode']):
    """Internal memory blocks of the TPFlock."""

    def __init__(self, owner):
        super().__init__(owner)
        self.tp = TPFlockInternalsSection(self)

    def prepare_slots(self, unit: TemporalPoolerFlockUnit):
        self.tp.prepare_slots_from_flock(unit.flock, None, self._owner.params.flock_size)


class TemporalPoolerFlockNode(WorkerNodeWithInternalsBase[TPFlockInputs, TPFlockInternals, TPFlockOutputs],
                              LearningSwitchable):
    """Node containing one flock of temporal poolers only.

    From the three inputs, only data_input is mandatory, rest can stay unconnected.
    """

    _unit: TemporalPoolerFlockUnit
    inputs: TPFlockInputs
    memory_blocks: TPFlockInternals
    outputs: TPFlockOutputs
    _seed: int
    _tp_params_props: TemporalPoolerParamsProps

    def __init__(self, params: ExpertParams, seed: int = None, name="TPFlock"):
        super().__init__(name=name, inputs=TPFlockInputs(self), memory_blocks=TPFlockInternals(self),
                         outputs=TPFlockOutputs(self))
        self.params = params.clone()
        self._seed = seed
        self._create_params_props()

    def _create_unit(self, creator: TensorCreator) -> Unit:
        set_global_seeds(self._seed)
        return TemporalPoolerFlockUnit(creator, self.params)

    def _step(self):
        self._unit.step(self.inputs.tp.data_input.tensor,
                        self.inputs.tp.context_input.tensor,
                        self.inputs.tp.reward_input.tensor)

    def _on_initialization_change(self):
        self._create_params_props()

    def _create_params_props(self):
        self._tp_params_props = TemporalPoolerParamsProps(self.params.temporal,
                                                          self._unit.flock if self._unit is not None else None)

    def get_properties(self):
        return self._tp_params_props.get_properties()

    def switch_learning(self, on: bool):
        self._tp_params_props.enable_learning = on

    def _get_process_tp_trained_forward(self) -> Optional[TPFlockForwardAndBackward]:
        return None if self._unit is None else self._unit.flock.trained_forward_process

    def _get_process_tp_learn(self) -> Optional[TPFlockLearning]:
        return None if self._unit is None else self._unit.flock.learn_process

    def _get_observables(self) -> Dict[str, Observable]:
        result = super()._get_observables()

        flock_size = self.params.flock_size

        result.update(create_tp_forward_observables(flock_size, self._get_process_tp_trained_forward))
        result.update(create_tp_learn_observables(flock_size, self._get_process_tp_learn))

        return result
