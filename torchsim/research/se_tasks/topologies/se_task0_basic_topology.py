from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import UnsqueezeNode, SpatialPoolerFlockNode
from torchsim.research.se_tasks.topologies.se_task0_topology import SeT0TopologicalGraph


class SeT0BasicTopology(SeT0TopologicalGraph):
    """A model which receives data from the 0th SE task and learns spatial and temporal patterns (task0)."""
    def _install_experts(self):
        self._top_level_flock_node = SpatialPoolerFlockNode(self._create_expert_params())
        self.add_node(self._top_level_flock_node)
        self.unsqueeze_node = UnsqueezeNode(0)
        self.add_node(self.unsqueeze_node)
        Connector.connect(self.se_io.outputs.image_output, self._join_node.inputs[0])
        Connector.connect(self.se_io.outputs.task_to_agent_label, self._join_node.inputs[1])
        Connector.connect(self._join_node.outputs[0], self.unsqueeze_node.inputs.input)
        Connector.connect(self.unsqueeze_node.outputs.output, self._top_level_flock_node.inputs.sp.data_input)

    def _get_agent_output(self):
        return self._top_level_flock_node.outputs.sp.current_reconstructed_input

    def _top_level_expert_output_size(self):
        return self.se_io.get_image_numel()

    @staticmethod
    def _create_expert_params() -> ExpertParams:
        expert_params = ExpertParams()
        expert_params.flock_size = 1
        expert_params.n_cluster_centers = 200
        expert_params.compute_reconstruction = True
        expert_params.spatial.batch_size = 1000
        expert_params.spatial.buffer_size = 1010
        expert_params.spatial.cluster_boost_threshold = 200
        return expert_params
