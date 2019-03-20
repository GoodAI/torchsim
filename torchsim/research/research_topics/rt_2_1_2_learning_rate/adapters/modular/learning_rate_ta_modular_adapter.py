from typing import Any, List

import torch
import logging

from torchsim.core.eval.metrics.sp_convergence_metrics import average_boosting_duration, num_boosted_clusters
from torchsim.core.eval.node_accessors.flock_node_accessor import FlockNodeAccessor
from torchsim.core.eval.node_accessors.sp_node_accessor import SpatialPoolerFlockNodeAccessor
from torchsim.core.nodes import ExpertFlockNode, ConvExpertFlockNode, SpatialPoolerFlockNode, ConvSpatialPoolerFlockNode
from torchsim.research.experiment_templates.task0_train_test_learning_rate_template import \
    TaTask0TrainTestClassificationAccAdapter
from torchsim.research.research_topics.rt_2_1_2_learning_rate.adapters.modular.classification_accuracy_modular_adapter import \
    ClassificationAccuracyModularAdapter

logger = logging.getLogger(__name__)


class LearningRateTaModularAdapter(ClassificationAccuracyModularAdapter, TaTask0TrainTestClassificationAccAdapter):

    _flock_nodes: List[Any]  # List of experts ~ layers

    def set_topology(self, topology):
        super().set_topology(topology)

        # convert conv_layers and top_expert into list of layers (backward compatibility)
        layers = self._model.conv_layers
        self._flock_nodes = []

        for layer in layers:
            self._flock_nodes.append(layer.expert_flock_nodes[0])

        self._flock_nodes.append(self._model.top_layer.sp_node)

    def get_flock_size_of(self, layer_id: int) -> int:
        return self._flock_nodes[layer_id].params.flock_size

    def get_output_id_for(self, layer_id: int) -> int:
        return FlockNodeAccessor.get_sp_output_id(self._flock_nodes[layer_id])

    def get_baseline_output_id_for(self, layer_id: int) -> int:
        return self._se_node_group.get_baseline_output_id_for(layer_id)

    def clone_sp_output_tensor_for(self, layer_id: int) -> torch.Tensor:
        flock = self._flock_nodes[layer_id]
        flock_size = self.get_flock_size_of(layer_id)

        if type(flock) is ExpertFlockNode or type(flock) is ConvExpertFlockNode:
            # turns out that the conv expert can have shape [2,2,num_cc]
            return FlockNodeAccessor.get_sp_output_tensor(flock).clone().view(flock_size, -1)
        if type(flock) is SpatialPoolerFlockNode or type(flock) is ConvSpatialPoolerFlockNode:
            return SpatialPoolerFlockNodeAccessor.get_output_tensor(flock).clone().view(flock_size, -1)

        logger.error(f'unsupported flock class type')

    def get_sp_size_for(self, layer_id: int) -> int:
        return self.clone_sp_output_tensor_for(layer_id).numel()

    def get_average_boosting_duration_for(self, layer_id: int) -> float:
        return average_boosting_duration(
            FlockNodeAccessor.get_sp_boosting_durations(self._flock_nodes[layer_id]))

    def get_num_boosted_clusters_ratio(self, layer_id: int) -> float:
        return num_boosted_clusters(
            FlockNodeAccessor.get_sp_boosting_durations(self._flock_nodes[layer_id]))
