import logging
from dataclasses import dataclass
from typing import List

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.node_group import NodeGroupBase, GroupInputs, GroupOutputs
from torchsim.core.nodes import UnsqueezeNode, ScatterNode, SwitchNode, JoinNode
from torchsim.research.research_topics.rt_4_1_1_gradual_learning_basic.topologies.dataset_alphabet_node_group import \
    DatasetAlphabetNodeGroupParams, DatasetAlphabetNodeGroup
from torchsim.research.research_topics.rt_4_1_1_gradual_learning_basic.topologies.flock_partial_switch_node_group import \
    FlockPartialSwitchNodeGroup, FlockPartialSwitchNodeGroupParams

logger = logging.getLogger(__name__)


@dataclass
class DatasetSwitchNodeGroupParams:
    dataset_params: DatasetAlphabetNodeGroupParams
    flock_split: int = 0


class DatasetSwitchOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")
        self.sequence_id_one_hot = self.create("Sequence ID one hot")
        self.dataset_2_scalar_sequence_ids = self.create("Dataset 2 sequence ID")


class DatasetSwitchNodeGroup(NodeGroupBase[EmptyInputs, DatasetSwitchOutputs]):
    _switches: List[FlockPartialSwitchNodeGroup]
    _datasets: List[DatasetAlphabetNodeGroup]

    def __init__(self, params: DatasetSwitchNodeGroupParams, name: str = "DatasetSwitchNodeGroup", ):
        super().__init__(name, inputs=EmptyInputs(self), outputs=DatasetSwitchOutputs(self))
        self._params = params

        n_dataset_1 = DatasetAlphabetNodeGroup(params.dataset_params)
        n_dataset_2 = DatasetAlphabetNodeGroup(params.dataset_params)

        n_switch_output = self.create_flock_switch("DatasetSwitch output")
        n_switch_seq_id = self.create_flock_switch("DatasetSwitch seq_id")

        # n_switch_output = SwitchNode(2, name='DatasetSwitch output')
        # n_switch_seq_id = SwitchNode(2, name='DatasetSwitch seq_id')
        # n_join_node = JoinNode(dim=0, n_inputs=2)
        self.add_node(n_dataset_1)
        self.add_node(n_dataset_2)
        self.add_node(n_switch_output)
        self.add_node(n_switch_seq_id)

        Connector.connect(n_dataset_1.outputs.output, n_switch_output.inputs.input_1)
        Connector.connect(n_dataset_2.outputs.output, n_switch_output.inputs.input_2)

        Connector.connect(n_dataset_1.outputs.sequence_id_one_hot, n_switch_seq_id.inputs.input_1)
        Connector.connect(n_dataset_2.outputs.sequence_id_one_hot, n_switch_seq_id.inputs.input_2)

        Connector.connect(n_switch_output.outputs.output, self.outputs.output.input)
        Connector.connect(n_switch_seq_id.outputs.output, self.outputs.sequence_id_one_hot.input)

        self._datasets = [n_dataset_1, n_dataset_2]
        self._switches = [n_switch_output, n_switch_seq_id]

        Connector.connect(n_dataset_2.outputs.scalar_sequence_ids, self.outputs.dataset_2_scalar_sequence_ids.input)

    def create_flock_switch(self, name: str) -> FlockPartialSwitchNodeGroup:
        return FlockPartialSwitchNodeGroup(FlockPartialSwitchNodeGroupParams(
            flock_size=self._params.dataset_params.flock_size,
            split_idx=self._params.flock_split
        ), name=name)

    def set_sequences_filter(self, dataset_id: int, enabled_sequences: List[bool]):
        self._datasets[dataset_id].set_sequences_filter(enabled_sequences)

    def select_dataset(self, dataset_id: int):
        for n_switch in self._switches:
            n_switch.set_active_input_index(dataset_id)

    def init_sp_clusters(self):
        for dataset in self._datasets:
            dataset.init_sp_clusters()
