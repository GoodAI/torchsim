from torchsim.core.eval.node_accessors.random_number_accessor import RandomNumberNodeAccessor
from torchsim.core.eval.topology_adapter_base import TestableTopologyAdapterBase
from torchsim.core.graph import Topology
from torchsim.core.nodes.random_number_node import RandomNumberNode
from torchsim.topologies.random_number_topology import RandomNumberTopology


class RandomNumberTopologyTrainTestAdapter(TestableTopologyAdapterBase):
    _is_training: bool = True
    _topology: RandomNumberTopology
    _node: RandomNumberNode

    # def __init__(self):
    #     self._is_training = True

    def is_in_training_phase(self, **kwargs) -> bool:
        return self._is_training

    def switch_to_training(self):
        self._is_training = True

    def switch_to_testing(self):
        self._is_training = False

    def get_topology(self) -> Topology:
        return self._topology

    def set_topology(self, topology: RandomNumberTopology):
        self._topology = topology
        self._node = topology.random_number_node

    def get_output_id(self):
        return RandomNumberNodeAccessor.get_output_id(self._node)