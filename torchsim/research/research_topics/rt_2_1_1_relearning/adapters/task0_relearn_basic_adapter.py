import math

import torch

from torchsim.core.eval.metrics.sp_convergence_metrics import average_sp_delta, average_boosting_duration
from torchsim.core.eval.node_accessors.random_number_accessor import RandomNumberNodeAccessor
from torchsim.core.eval.node_accessors.se_io_accessor import SeIoAccessor
from torchsim.core.eval.node_accessors.sp_node_accessor import SpatialPoolerFlockNodeAccessor
from torchsim.core.graph import Topology
from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode
from torchsim.research.experiment_templates.task0_train_test_template_relearning import AbstractRelearnAdapter

from torchsim.research.research_topics.rt_2_1_1_relearning.topologies.task0_basic_topology import SeT0BasicTopologyRT211
from torchsim.research.research_topics.rt_2_1_1_relearning.topologies.task0_basic_topology_phased import \
    SeT0BasicTopologyRT211Phased
from torchsim.research.se_tasks.topologies.se_io.se_io_task0_dataset import SeIoTask0Dataset
from torchsim.research.se_tasks.topologies.se_io.se_io_task0_dataset_phased import SeIoTask0DatasetPhased


class Task0RelearnBasicAdapter(AbstractRelearnAdapter):
    _topology: SeT0BasicTopologyRT211Phased

    def get_sp_output_id(self) -> int:
        return self.sp_output_tensor.argmax().item()

    def get_label_id(self) -> int:
        return SeIoAccessor.get_label_id(self.se_io)

    def get_sp_size(self) -> int:
        return self.sp_output_tensor.numel()

    def clone_random_baseline_output_tensor_for_labels(self) -> torch.Tensor:
        return RandomNumberNodeAccessor.get_output_tensor(self._topology._random_label_baseline).clone()

    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        return self._topology._fork_node.outputs[1].tensor.clone()

    def get_average_log_delta(self) -> float:
        """Supports getting deltas just for the layer 0"""
        delta = average_sp_delta(
            SpatialPoolerFlockNodeAccessor.get_sp_deltas(self.flock_node))

        if delta > 0:  # avoid math.domain error
            delta = math.log(delta)
        return delta

    def get_average_boosting_duration(self) -> float:
        return average_boosting_duration(
            SpatialPoolerFlockNodeAccessor.get_sp_boosting_durations(self.flock_node))

    def get_topology(self) -> Topology:
        return self._topology

    def set_topology(self, topology: SeT0BasicTopologyRT211):
        self._topology = topology

    def is_in_training_phase(self, **kwargs) -> bool:
        return not self.topology.is_in_testing_phase()

    def switch_to_training(self):
        self.topology.switch_learning(True)

        # SE probably do not support manual switching between train/test
        assert type(self.se_io) is SeIoTask0DatasetPhased
        io_dataset: SeIoTask0Dataset = self.se_io
        io_dataset.node_se_dataset.switch_training(True, False)

    def switch_to_testing(self):
        self.topology.switch_learning(False)

        # SE probably do not support manual switching between train/test
        assert type(self.se_io) is SeIoTask0DatasetPhased
        io_dataset: SeIoTask0Dataset = self.se_io
        io_dataset.node_se_dataset.switch_training(False, False)

    @property
    def se_io(self) -> SeIoTask0Dataset:
        return self.topology.se_io

    @property
    def topology(self) -> SeT0BasicTopologyRT211Phased:
        return self._topology

    @property
    def flock_node(self)-> SpatialPoolerFlockNode:
        return self.topology._top_level_flock_node

    @property
    def sp_output_tensor(self) -> torch.Tensor:
        return SpatialPoolerFlockNodeAccessor.get_output_tensor(self.flock_node)
