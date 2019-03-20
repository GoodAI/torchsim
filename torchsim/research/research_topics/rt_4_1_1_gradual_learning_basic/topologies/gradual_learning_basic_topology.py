import torch

from dataclasses import dataclass

from functools import partial

from typing import List, Type, Any

from torchsim.core.graph import logging, Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams, SpatialPoolerParams
from torchsim.core.nodes import SpatialPoolerFlockNode, PassNode, JoinNode, RandomNoiseNode, RandomNoiseParams, ForkNode
from torchsim.core.nodes.accuracy_node import AccuracyNode
from torchsim.gui.observables import ObserverPropertiesItem, ObserverPropertiesBuilder, ObserverPropertiesItemSourceType
from torchsim.research.research_topics.rt_4_1_1_gradual_learning_basic.topologies.dataset_alphabet_node_group import \
    DatasetAlphabetNodeGroupParams
from torchsim.research.research_topics.rt_4_1_1_gradual_learning_basic.topologies.dataset_switch_node_group import \
    DatasetSwitchNodeGroup, DatasetSwitchNodeGroupParams
from torchsim.research.research_topics.rt_4_1_1_gradual_learning_basic.topologies.sp_format_context_node_group import \
    SPFormatContextNodeGroup
from torchsim.research.research_topics.rt_4_1_1_gradual_learning_basic.topologies.specialist_node_group import \
    SpecialistNodeGroup, SpecialistNodeGroupParams

logger = logging.getLogger(__name__)


@dataclass
class NotForgettingExperimentParams:
    phase_1_steps: int = 4000  # Learning of
    phase_2_steps: int = 2000
    phase_3_steps: int = 4000


@dataclass
class GLExperimentParams:
    component: Type = None
    params: Any = None


@dataclass
class GradualLearningBasicTopologyParams:
    gate_input_context_multiplier: float = 4.0
    gate_input_context_avg_window_size: int = 5
    gate_buffer_size: int = 1500
    symbols: str = "abcdefghijklmnopqrstuvwxyz0123456789"
    flock_size: int = 20
    flock_split: int = 5
    seq_count: int = 3
    seq_length: int = 10
    seq_repeat: int = 3
    convert_context_to_one_hot: bool = True
    accuracy_average_steps: int = 30 * 6
    experiment_params: GLExperimentParams = None


class GradualLearningBasicTopology(Topology):
    """
    Long words utilizing context

    Interesting observers:
      gate:
        * SP Learn Process - Data Batch, sum over dim 1 (zero values means sequence not present in batch)
        * SP cluster centers
        * SP output forward clusters
      specialist:
        * SP_frequent_seqs_reconstruction - symbols reconstruction
        * TP_frequent_context_likelihood - show context per each symbol in learnt sequences(items per row 2)
        * TP_seq_likelihoods_by_cluster
    """
    _n_accuracy_2: AccuracyNode
    _n_accuracy_1: AccuracyNode
    _n_accuracy_single_2: AccuracyNode
    _n_accuracy_single_1: AccuracyNode
    _n_dataset_switch: DatasetSwitchNodeGroup
    _n_specialist: SpecialistNodeGroup
    _prop_builder: ObserverPropertiesBuilder
    _step_count: int = 0
    _active_dataset: int = 0

    def __init__(self, params: GradualLearningBasicTopologyParams = GradualLearningBasicTopologyParams()):
        super().__init__('cuda')
        self._prop_builder = ObserverPropertiesBuilder(self, source_type=ObserverPropertiesItemSourceType.MODEL)

        self._params = params
        self.create_topology()

    @property
    def params(self):
        return self._params

    def create_topology(self):
        """
                                        +----------------+
            +-------------+             | dataset_switch |
            |             |             +--+-----+-------+
            |             v                |     |
            |  +----------+------------+   |     |
            |  | context_feedback_pass |   |     |
            |  +--------------------+--+   |     |
            |                       |      |     |
            |                       v      v     |
            |               +-------+------+--+  |
            |               | gate_input_join |  |
            |               +-------+---------+  |
            |                       |            |
            |                       v            |
            |              +--------+---------+  |
            |              | gate_input_noise |  |
            |              +--------+---------+  |
            |                       |            |
            |                       v            |
            |                   +---+--+         |
            |                   | gate |         |
            |                   +---+--+         |
            |                       |            |
            |                       v            |
            |               +-------+--------+   +--------+
            |               | format_context |   |        |
            |               +-------+--------+   |        |
            |                       |            v        |
            |                       |     +------+-----+  |
            |                       ---->-+ specialist |  |
            |                             +--+--------++  |
            |                                |        |   |
            +--------------------------------+        v   v
                                                   ++--------++
                                                   | accuracy |
                                                   +----------+
        """

        n_gate = SpatialPoolerFlockNode(
            ExpertParams(flock_size=self._params.flock_size,
                         n_cluster_centers=self._params.seq_count,
                         spatial=SpatialPoolerParams(
                             # input_size=3,
                             enable_learning=True,
                             buffer_size=self._params.gate_buffer_size,
                             batch_size=100,
                             learning_rate=0.2,
                             learning_period=10,
                             cluster_boost_threshold=100,
                             max_boost_time=200
                         ),
                         ),
            name="Gate"
        )
        self.add_node(n_gate)

        # Specialist
        n_specialist = SpecialistNodeGroup(SpecialistNodeGroupParams(
            flock_size=self._params.flock_size,
            n_symbols=len(self._params.symbols),
            gate_input_context_multiplier=self._params.gate_input_context_multiplier,
            gate_input_context_avg_window_size=self._params.gate_input_context_avg_window_size,
            seq_count=self._params.seq_count,
            convert_context_to_one_hot=self._params.convert_context_to_one_hot
        ))
        self.add_node(n_specialist)
        self._n_specialist = n_specialist

        n_context_feedback_pass = PassNode((self._params.flock_size, self._params.seq_count))
        n_gate_input_join = JoinNode(dim=1, n_inputs=2)
        n_gate_input_noise = RandomNoiseNode(RandomNoiseParams(amplitude=0.0001))
        n_format_context = SPFormatContextNodeGroup(self._params.seq_count, self._params.flock_size)

        self.add_node(n_context_feedback_pass)
        self.add_node(n_gate_input_join)
        self.add_node(n_gate_input_noise)
        self.add_node(n_format_context)

        # Dataset
        n_dataset_switch = DatasetSwitchNodeGroup(DatasetSwitchNodeGroupParams(
            dataset_params=DatasetAlphabetNodeGroupParams(
                flock_size=self._params.flock_size,
                symbols=self._params.symbols,
                seq_length=self._params.seq_length,
                seq_count=self._params.seq_count,
                seq_repeat=self._params.seq_repeat
            ),
            flock_split=self._params.flock_split
        ))

        self._n_dataset_switch = n_dataset_switch
        self.add_node(n_dataset_switch)

        # dataset to specialist
        Connector.connect(n_dataset_switch.outputs.output, n_specialist.inputs.input)
        # specialist to gate
        Connector.connect(n_specialist.outputs.context_feedback, n_context_feedback_pass.inputs.input, is_backward=True)
        Connector.connect(n_context_feedback_pass.outputs.output, n_gate_input_join.inputs[0])
        # dataset to gate
        Connector.connect(n_dataset_switch.outputs.sequence_id_one_hot, n_gate_input_join.inputs[1])
        Connector.connect(n_gate_input_join.outputs.output, n_gate_input_noise.inputs.input)
        Connector.connect(n_gate_input_noise.outputs.output, n_gate.inputs.sp.data_input)
        # gate to specialist
        Connector.connect(n_gate.outputs.sp.forward_clusters, n_format_context.inputs.input)
        Connector.connect(n_format_context.outputs.output, n_specialist.inputs.context_input)

        # Measuring accuracy
        # Fork
        n_fork_dataset = ForkNode(0, [self._params.flock_split, self._params.flock_size - self._params.flock_split])
        n_fork_prediction = ForkNode(0, [self._params.flock_split, self._params.flock_size - self._params.flock_split])
        self.add_node(n_fork_dataset)
        self.add_node(n_fork_prediction)
        Connector.connect(n_dataset_switch.outputs.output, n_fork_dataset.inputs.input)
        Connector.connect(n_specialist.outputs.output, n_fork_prediction.inputs.input)

        self._n_accuracy_single_1 = AccuracyNode(1, name='Accuracy single 1')
        self.add_node(self._n_accuracy_single_1)
        Connector.connect(n_fork_dataset.outputs[0], self._n_accuracy_single_1.inputs.input_a)
        Connector.connect(n_fork_prediction.outputs[0], self._n_accuracy_single_1.inputs.input_b)

        self._n_accuracy_single_2 = AccuracyNode(1, name='Accuracy single 2')
        self.add_node(self._n_accuracy_single_2)
        Connector.connect(n_fork_dataset.outputs[1], self._n_accuracy_single_2.inputs.input_a)
        Connector.connect(n_fork_prediction.outputs[1], self._n_accuracy_single_2.inputs.input_b)

        self._n_accuracy_1 = AccuracyNode(self._params.accuracy_average_steps, name='Accuracy 1')
        self.add_node(self._n_accuracy_1)
        Connector.connect(n_fork_dataset.outputs[0], self._n_accuracy_1.inputs.input_a)
        Connector.connect(n_fork_prediction.outputs[0], self._n_accuracy_1.inputs.input_b)

        self._n_accuracy_2 = AccuracyNode(self._params.accuracy_average_steps, name='Accuracy 2')
        self.add_node(self._n_accuracy_2)
        Connector.connect(n_fork_dataset.outputs[1], self._n_accuracy_2.inputs.input_a)
        Connector.connect(n_fork_prediction.outputs[1], self._n_accuracy_2.inputs.input_b)

    def init_sp_clusters(self):
        self._n_dataset_switch.init_sp_clusters()
        self._n_specialist.init_sp_clusters()

    def set_sequences_filter(self, dataset_id: int, enabled_sequences: List[bool]):
        self._n_dataset_switch.set_sequences_filter(dataset_id, enabled_sequences)
        logger.info(f'sequence filter: {enabled_sequences}, step: {self._step_count}')

    @property
    def active_dataset(self) -> int:
        return self._active_dataset

    @active_dataset.setter
    def active_dataset(self, value: int):
        self._active_dataset = value
        self._n_dataset_switch.select_dataset(value)
        logger.info(f'active dataset: {value}, step: {self._step_count}')

    def get_properties(self) -> List[ObserverPropertiesItem]:
        props = super().get_properties()
        return props + [
            self._prop_builder.collapsible_header(f'Experiment', True),
            self._prop_builder.auto("Active dataset", type(self).active_dataset),
            *self._dataset_controll_buttons(0),
            *self._dataset_controll_buttons(1)
        ]

    def _dataset_controll_buttons(self, dataset_id: int) -> List[ObserverPropertiesItem]:
        patterns = [
            [False, False, False] * 2,
            [True, False, False] * 2,
            [False, True, False] * 2,
            [False, False, True] * 2,
            [True, True, False] * 2,
            [False, True, True] * 2,
            [True, False, True] * 2,
            [True, True, True] * 2,
            [True, True, True, True, True, False],
        ]

        def format_pattern(pattern: List[bool]) -> str:
            return "".join(['1' if p else '0' for p in pattern])

        return [
            self._prop_builder.button(f'Dataset {dataset_id} - {format_pattern(p)}',
                                      partial(self.set_sequences_filter, dataset_id, p))
            for p in patterns
        ]

    def get_accuracy_single_1(self) -> float:
        return self._n_accuracy_single_1.outputs.accuracy.tensor.item()

    def get_accuracy_per_flock_single_1(self) -> List[float]:
        return self._n_accuracy_single_1.outputs.accuracy_per_flock.tensor.tolist()

    def get_accuracy_1(self) -> float:
        return self._n_accuracy_1.outputs.accuracy.tensor.item()

    def get_accuracy_per_flock_1(self) -> List[float]:
        return self._n_accuracy_1.outputs.accuracy_per_flock.tensor.tolist()

    def get_accuracy_single_2(self) -> float:
        return self._n_accuracy_single_2.outputs.accuracy.tensor.item()

    def get_accuracy_per_flock_single_2(self) -> List[float]:
        return self._n_accuracy_single_2.outputs.accuracy_per_flock.tensor.tolist()

    def get_accuracy_2(self) -> float:
        return self._n_accuracy_2.outputs.accuracy.tensor.item()

    def get_accuracy_per_flock_2(self) -> List[float]:
        return self._n_accuracy_2.outputs.accuracy_per_flock.tensor.tolist()

    def get_actual_sequence_ids(self) -> List[int]:
        return self._n_dataset_switch.outputs.dataset_2_scalar_sequence_ids.tensor.tolist()

    def step(self):
        super().step()
        self._step_count += 1
