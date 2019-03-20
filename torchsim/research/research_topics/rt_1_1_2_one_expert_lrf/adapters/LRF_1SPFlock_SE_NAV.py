import torch

from torchsim.core.graph import Topology
from torchsim.core.models.flock.flock_utils import memory_report
from torchsim.core.nodes.dataset_se_navigation_node import DatasetSeNavigationNode
from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode
from torchsim.research.experiment_templates.lrf_1sp_flock_template import Lrf1SpFlockTemplate
from torchsim.research.research_topics.rt_1_1_2_one_expert_lrf.topologies.se_nav_lrf_topology import SeNavLrfTopology


class Lrf1SpFlockSeNavTemplate(Lrf1SpFlockTemplate):
    _topology: SeNavLrfTopology
    # _mnist: DatasetMNISTNode
    _sp: SpatialPoolerFlockNode

    def set_topology(self, topology: SeNavLrfTopology):
        self._topology = topology
        # self._mnist = self._model.mnist_node
        self._sp = self._topology._node_spatial_pooler

    def get_topology(self) -> Topology:
        return self._topology

    def get_reconstructed_image(self) -> torch.Tensor:
        return self._topology.reconstructed_data.clone()

    def get_input_image(self) -> torch.Tensor:
        nav_node: DatasetSeNavigationNode = self._topology.se_nav_node
        return nav_node.outputs.image_output.tensor.clone()

    def get_difference_image(self) -> torch.Tensor:
        return self._topology.image_difference.clone()

    def get_memory_used(self) -> float:
        return memory_report(self._sp, printer=None)

    def get_label(self) -> float:
        nav_node: DatasetSeNavigationNode = self._topology.se_nav_node
        return nav_node.outputs.task_to_agent_location_int.tensor.clone()

    def get_sp_output(self) -> torch.Tensor:
        return self._sp.outputs.sp.forward_clusters.tensor.clone()

    def get_is_testing(self) -> bool:
        return self._topology.testing_phase

    def get_testing_phase_number(self) -> bool:
        return self._topology.n_testing_phase

    def get_training_step(self) -> int:
        return self._topology.training_step
