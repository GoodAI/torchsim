import logging
import math

import torch

from torchsim.core.eval.metrics.sp_convergence_metrics import average_sp_delta
from torchsim.core.eval.node_accessors.flock_node_accessor import FlockNodeAccessor
from torchsim.core.eval.node_accessors.sp_node_accessor import SpatialPoolerFlockNodeAccessor
from torchsim.core.nodes import ExpertFlockNode, ConvExpertFlockNode, SpatialPoolerFlockNode, \
    ConvSpatialPoolerFlockNode
from torchsim.research.research_topics.rt_2_1_2_learning_rate.adapters.modular.model_classification_adapter_base import \
    ModelClassificationAdapterBase
from torchsim.topologies.toyarch_groups.ncmr1_group import NCMR1Group

logger = logging.getLogger(__name__)


class Nc1r1GroupWithAdapter(NCMR1Group, ModelClassificationAdapterBase):

    # adapter purposes
    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        return self.top_layer.outputs.label.tensor.clone()

    def get_average_log_delta_for(self, layer_id: int) -> float:
        """Supports getting deltas just for the layer 0"""
        assert layer_id <= len(self.conv_layers)

        if layer_id == len(self.conv_layers):
            expert = self.top_layer.sp_node
        else:
            expert = self.conv_layers[layer_id].expert_flock_nodes[0]

        deltas = None
        if type(expert) is ExpertFlockNode or type(expert) is ConvExpertFlockNode:
            # turns out that the conv expert can have shape [2,2,num_cc]
            deltas = FlockNodeAccessor.get_sp_deltas(expert)
        elif type(expert) is SpatialPoolerFlockNode or type(expert) is ConvSpatialPoolerFlockNode:
            deltas = SpatialPoolerFlockNodeAccessor.get_sp_deltas(expert)
        else:
            logger.error("unsupported expert type!")

        delta = average_sp_delta(deltas)

        if delta > 0:  # avoid math.domain error
            delta = math.log(delta)
        return delta

    def model_switch_to_training(self):
        self._switch_training(True)

    def model_switch_to_testing(self):
        self._switch_training(False)

    def _switch_training(self, training: bool):
        logger.info(f'Topology: changing the learning state to learning_on: {training}')
        try:
            for layer in self.conv_layers:
                layer.switch_learning(training)
            self.top_layer.switch_learning(training)

            self._is_learning = training
            logger.info(f'switching learning to {training} ')
        except:
            logger.info(f'unable to switch the learning to {training}')

    def model_is_learning(self):
        return self._is_learning
