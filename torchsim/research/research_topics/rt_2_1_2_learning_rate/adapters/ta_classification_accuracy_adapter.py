import math
from typing import List

import torch

from torchsim.core.eval.metrics.sp_convergence_metrics import average_sp_delta
from torchsim.core.eval.node_accessors.flock_node_accessor import FlockNodeAccessor
from torchsim.core.nodes import ForkNode, ExpertFlockNode
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.conv_wide_two_layer_topology import \
    ConvWideTwoLayerTopology
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.task0_base_topology import Task0BaseTopology
from torchsim.research.research_topics.rt_2_1_2_learning_rate.adapters.classification_accuracy_adapter import \
    ClassificationAccuracyAdapterBase


class TaClassificationAccuracyAdapter(ClassificationAccuracyAdapterBase):
    """Compute classification accuracy of the TA on the test set"""

    _topology: ConvWideTwoLayerTopology
    _flock_nodes: List[ExpertFlockNode]  # layers of flocks
    _fork_node: ForkNode

    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        return self._fork_node.outputs[1].tensor.clone()

    def get_average_log_delta_for(self, layer_id: int) -> float:
        """Supports getting deltas just for the layer 0"""
        delta = average_sp_delta(
            FlockNodeAccessor.get_sp_deltas(self._flock_nodes[layer_id]))

        if delta > 0:  # avoid math.domain error
            delta = math.log(delta)
        return delta

    def set_topology(self, topology: Task0BaseTopology):
        super().set_topology(topology)

        self._flock_nodes = self._topology.flock_nodes
        self._fork_node = self._topology.fork_node

    def model_switch_to_training(self):
        self._topology.switch_learning(True)

    def model_switch_to_testing(self):
        self._topology.switch_learning(False)

    def is_learning(self) -> bool:
        return self._topology.is_learning
