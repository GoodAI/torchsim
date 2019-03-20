import logging
from typing import List

import torch

from torchsim.core.eval.metrics.sp_convergence_metrics import average_boosting_duration
from torchsim.core.eval.node_accessors.flock_node_accessor import FlockNodeAccessor
from torchsim.core.eval.node_accessors.random_number_accessor import RandomNumberNodeAccessor
from torchsim.core.eval.node_accessors.sp_node_accessor import SpatialPoolerFlockNodeAccessor
from torchsim.core.nodes.conv_expert_node import ConvExpertFlockNode
from torchsim.core.nodes.expert_node import ExpertFlockNode
from torchsim.core.nodes.random_number_node import RandomNumberNode
from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode
from torchsim.research.experiment_templates.task0_train_test_learning_rate_template import \
    TaTask0TrainTestClassificationAccAdapter
from torchsim.research.research_topics.rt_2_1_2_learning_rate.adapters.ta_classification_accuracy_adapter import \
    TaClassificationAccuracyAdapter

logger = logging.getLogger(__name__)


class Task0TaMultilayerAdapter(TaTask0TrainTestClassificationAccAdapter, TaClassificationAccuracyAdapter):

    _baselines: List[RandomNumberNode]

    def get_flock_size_of(self, layer_id: int) -> int:
        return self._flock_nodes[layer_id].params.flock_size

    def get_output_id_for(self, layer_id: int) -> int:
        return FlockNodeAccessor.get_sp_output_id(self._flock_nodes[layer_id])

    def set_topology(self, topology):
        super().set_topology(topology)
        self._baselines = self._topology.baselines

    def get_baseline_output_id_for(self, layer_id: int) -> int:
        return RandomNumberNodeAccessor.get_output_id(self._baselines[layer_id])

    def clone_sp_output_tensor_for(self, layer_id: int) -> torch.Tensor:
        flock = self._flock_nodes[layer_id]
        flock_size = self.get_flock_size_of(layer_id)

        if type(flock) is ExpertFlockNode or type(flock) is ConvExpertFlockNode:
            # turns out that the conv expert can have shape [2,2,num_cc]
            flock: ExpertFlockNode = flock
            return FlockNodeAccessor.get_sp_output_tensor(flock).clone().view(flock_size, -1)
        if type(flock) is SpatialPoolerFlockNode:
            flock: SpatialPoolerFlockNode = flock
            return SpatialPoolerFlockNodeAccessor.get_output_tensor(flock).clone().view(flock_size, -1)

        logger.error(f'unsupported flock class type')

    def get_sp_size_for(self, layer_id: int) -> int:
        return self.clone_sp_output_tensor_for(layer_id).numel()

    # TODO get device from topology somehow
    # def get_device(self) -> str:
    #     self._baselines[0].devi

    def get_average_boosting_duration_for(self, layer_id: int) -> float:
        return average_boosting_duration(
            FlockNodeAccessor.get_sp_boosting_durations(self._flock_nodes[layer_id]))