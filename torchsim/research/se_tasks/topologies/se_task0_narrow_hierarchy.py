from typing import List

from torchsim.core.graph.connection import Connector
from torchsim.core.nodes import ExpertFlockNode, SpatialPoolerFlockNode, UnsqueezeNode
from torchsim.research.se_tasks.topologies.se_task0_topology import SeT0TopologicalGraph


class SeT0NarrowHierarchy(SeT0TopologicalGraph):
    _N_LEVEL_1_CLUSTER_CENTERS: int = 20
    _N_LEVEL_1_FLOCKS: int = 1
    _flock_nodes: List[ExpertFlockNode]

    def _install_experts(self):
        self._top_level_flock_node = SpatialPoolerFlockNode(self._create_expert_params())
        self._flock_nodes = [ExpertFlockNode(self._create_expert_params()),
                             ExpertFlockNode(self._create_expert_params()),
                             self._top_level_flock_node]
        for node in self._flock_nodes:
            self.add_node(node)

        unsqueeze_node_0 = UnsqueezeNode(0)
        self.add_node(unsqueeze_node_0)

        unsqueeze_node_1 = UnsqueezeNode(0)
        self.add_node(unsqueeze_node_1)

        Connector.connect(self.se_io.outputs.image_output, unsqueeze_node_0.inputs.input)
        Connector.connect(unsqueeze_node_0.outputs.output, self._flock_nodes[0].inputs.sp.data_input)
        Connector.connect(self._flock_nodes[0].outputs.tp.projection_outputs, self._flock_nodes[1].inputs.sp.data_input)
        Connector.connect(self._flock_nodes[1].outputs.tp.projection_outputs, self._join_node.inputs[0])
        Connector.connect(self.se_io.outputs.task_to_agent_label, self._join_node.inputs[1])
        Connector.connect(self._join_node.outputs.output, unsqueeze_node_1.inputs.input)
        Connector.connect(unsqueeze_node_1.outputs.output, self._top_level_flock_node.inputs.sp.data_input)

    def _get_agent_output(self):
        return self._top_level_flock_node.outputs.sp.current_reconstructed_input

    def _top_level_expert_output_size(self):
        top_level_expert_input_size = self._N_LEVEL_1_CLUSTER_CENTERS * self._N_LEVEL_1_FLOCKS
        return top_level_expert_input_size  # We output the reconstructed input
