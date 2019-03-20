import logging
import math
from typing import Optional, Tuple

import torch

from torchsim.core.eval.metrics.sp_convergence_metrics import average_sp_delta
from torchsim.core.eval.node_accessors.flock_node_accessor import FlockNodeAccessor
from torchsim.core.eval.node_accessors.sp_node_accessor import SpatialPoolerFlockNodeAccessor
from torchsim.core.graph.connection import Connector
from torchsim.core.nodes import ExpertFlockNode, ConvExpertFlockNode, SpatialPoolerFlockNode, \
    ConvSpatialPoolerFlockNode
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.research.research_topics.rt_2_1_2_learning_rate2.node_groups.classification_model_group import \
    ClassificationModelGroup
from torchsim.topologies.toyarch_groups.ncmr1_group import NCMR1Group
from torchsim.topologies.toyarch_groups.ncm_group import NCMGroup

logger = logging.getLogger(__name__)


class Nc1r1ClassificationGroup(ClassificationModelGroup):
    def __init__(self,
                 conv_layers_params: MultipleLayersParams,
                 top_layer_params: Optional[MultipleLayersParams] = None,
                 model_seed: int = 321,
                 num_labels: int = 20,
                 image_size: Tuple[int, int, int] = (24, 24, 3)):
        super().__init__("Nc1r1 Classification Group")

        if top_layer_params is None:
            self._group = self.add_node(NCMGroup(conv_layers_params,
                                                 model_seed,
                                                 image_size))
        else:
            self._group = self.add_node(NCMR1Group(conv_layers_params,
                                                   top_layer_params,
                                                   model_seed,
                                                   num_labels,
                                                   image_size))
            Connector.connect(self.inputs.label.output,
                              self._group.inputs.label)

        Connector.connect(self.inputs.image.output,
                          self._group.inputs.image)

    # adapter purposes
    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        return self._group.top_layer.outputs.label.tensor.clone()

    def get_average_log_delta_for(self, layer_id: int) -> float:
        """Supports getting deltas just for the layer 0"""
        assert layer_id <= len(self._group.conv_layers)

        if layer_id == len(self._group.conv_layers):
            expert = self._group.top_layer.sp_node
        else:
            expert = self._group.conv_layers[layer_id].expert_flock_nodes[0]

        if type(expert) is ExpertFlockNode or type(expert) is ConvExpertFlockNode:
            # turns out that the conv expert can have shape [2,2,num_cc]
            deltas = FlockNodeAccessor.get_sp_deltas(expert)
        elif type(expert) is SpatialPoolerFlockNode or type(expert) is ConvSpatialPoolerFlockNode:
            deltas = SpatialPoolerFlockNodeAccessor.get_sp_deltas(expert)
        else:
            logger.error("unsupported expert type!")
            return 0

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
            for layer in self._group.conv_layers:
                layer.switch_learning(training)
            self._group.top_layer.switch_learning(training)

            self._is_learning = training
            logger.info(f'switching learning to {training} ')
        except:
            logger.info(f'unable to switch the learning to {training}')

    def model_is_learning(self):
        return self._is_learning
