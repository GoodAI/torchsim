import math

from torchsim.core.eval.metrics.sp_convergence_metrics import average_sp_delta, average_boosting_duration
from torchsim.core.eval.node_accessors.random_number_accessor import RandomNumberNodeAccessor
from torchsim.core.eval.node_accessors.se_io_accessor import SeIoAccessor
from torchsim.core.eval.node_accessors.sp_node_accessor import SpatialPoolerFlockNodeAccessor
from torchsim.core.model import Model
from torchsim.core.nodes import random_number_node
from torchsim.core.nodes.dataset_se_navigation_node import DatasetSeNavigationNode
from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode
from torchsim.research.experiment_templates.sp_learning_convergence_template import SpLearningConvergenceTopologyAdapter
from torchsim.research.research_topics.rt_1_1_1_one_expert_sp.topologies.se_dataset_sp_topology import SeDatasetSpTopology


class SeDatasetSpLearningConvergenceTopologyAdapter(SpLearningConvergenceTopologyAdapter):
    """SeDatasetSpLearningConvergenceTopologyAdapter: contains NavigationSeDataset and SpatialPooler Node."""

    _topology: SeDatasetSpTopology
    _dataset: DatasetSeNavigationNode
    _sp: SpatialPoolerFlockNode
    _baseline: random_number_node

    def set_topology(self, topology: SeDatasetSpTopology):
        self._topology = topology
        self._dataset = topology.node_dataset
        self._sp = topology.node_sp
        self._baseline = topology.node_random
        return

    def get_label_id(self) -> int:
        return SeIoAccessor.get_landmark_id_int(self._dataset.outputs)

    def get_learned_model_output_id(self) -> int:
        return SpatialPoolerFlockNodeAccessor.get_output_id(self._sp)

    def get_baseline_output_id(self) -> int:
        return RandomNumberNodeAccessor.get_output_id(self._baseline)

    def get_title(self) -> str:
        return 'Mutual information of SP outputs and SE Dataset landmark labels'

    def get_topology(self) -> Model:
        return self._topology

    def get_device(self) -> str:
        return self._topology.device

    def get_model_output_size(self) -> int:
        return self._topology.output_dimension

    def get_average_delta(self) -> float:
        delta = average_sp_delta(SpatialPoolerFlockNodeAccessor.get_sp_deltas(self._sp))
        if delta > 0:  # avoid math.domain error
            delta = math.log(delta)
        return delta

    def get_average_boosting_duration(self) -> float:
        return average_boosting_duration(SpatialPoolerFlockNodeAccessor.get_sp_boosting_durations(self._sp))


