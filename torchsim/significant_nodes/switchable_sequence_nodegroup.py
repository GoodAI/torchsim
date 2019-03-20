from typing import Any, List

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.node_group import GroupOutputs, NodeGroupBase
from torchsim.core.nodes import SwitchNode, SequenceNode
from torchsim.core.utils.sequence_generator import SequenceGenerator, diagonal_transition_matrix


class SequenceOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner=owner)
        self.output = self.create("Sequence activation")
        self.sequence_num = self.create("Sequence number")


class SwitchableSequencesNodeGroup(NodeGroupBase[EmptyInputs, SequenceOutputs]):
    def __init__(self, sequence_generators: List[Any], name: str = "Switchable Sequences"):
        super().__init__(name, outputs=SequenceOutputs(self))
        self.sequences_len = len(sequence_generators)

        # switch
        switch_node = SwitchNode(self.sequences_len)
        self.switch_node = switch_node
        self.add_node(switch_node)

        switch_node_seq = SwitchNode(self.sequences_len)
        self.switch_node_seq = switch_node_seq
        self.add_node(switch_node_seq)

        self.env_nodes = []
        for i, sequence_generator in enumerate(sequence_generators):
            env_node = SequenceNode(sequence_generator, name=f"Sequence {i}")
            self.add_node(env_node)
            self.env_nodes.append(env_node)
            Connector.connect(env_node.outputs.output, switch_node.inputs[i])
            Connector.connect(env_node.outputs.sequence_num, switch_node_seq.inputs[i])

        Connector.connect(switch_node.outputs.output, self.outputs.output.input)
        Connector.connect(switch_node_seq.outputs.output, self.outputs.sequence_num.input)

    def switch_input(self):
        idx = (self.switch_node.active_input_index + 1) % self.sequences_len
        self.switch_input_to(idx)

    def switch_input_to(self, idx):
        for env_node in self.env_nodes:
            env_node.skip_execution = True
        self.env_nodes[idx].skip_execution = False
        self.switch_node.change_input(idx)
        self.switch_node_seq.change_input(idx)
