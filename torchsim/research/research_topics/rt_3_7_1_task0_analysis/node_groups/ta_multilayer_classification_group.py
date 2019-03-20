import logging
import math
from typing import Optional, Tuple, List, Union

import torch

from torchsim.core.eval.metrics.sp_convergence_metrics import average_sp_delta, average_boosting_duration, \
    num_boosted_clusters
from torchsim.core.eval.node_accessors.flock_node_accessor import FlockNodeAccessor
from torchsim.core.eval.node_accessors.sp_node_accessor import SpatialPoolerFlockNodeAccessor
from torchsim.core.graph.connection import Connector
from torchsim.core.nodes import ExpertFlockNode, ConvExpertFlockNode, SpatialPoolerFlockNode, \
    ConvSpatialPoolerFlockNode
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.node_groups.multilayer_model_group import MultilayerModelGroup
from torchsim.significant_nodes import SpReconstructionLayer
from torchsim.significant_nodes.conv_layer import ConvLayer
from torchsim.topologies.toyarch_groups.ncm_group import NCMGroup
from torchsim.topologies.toyarch_groups.ncmr1_group import NCMR1Group

logger = logging.getLogger(__name__)


class TaMultilayerClassificationGroup(MultilayerModelGroup):

    _top_layer: SpReconstructionLayer
    _conv_layers: List[ConvLayer]

    def __init__(self,
                 conv_layers_params: MultipleLayersParams,
                 top_layer_params: Optional[MultipleLayersParams] = None,
                 model_seed: int = 321,
                 num_labels: int = 20,
                 image_size: Tuple[int, int, int] = (24, 24, 3)):
        super().__init__("Nc1r1 Classification Group")

        if top_layer_params is None:
            # run just a nc1
            self._group = self.add_node(NCMGroup(conv_layers_params,
                                                 model_seed,
                                                 image_size))
        else:
            # run nc1 with reconstruction here
            self._group = self.add_node(NCMR1Group(conv_layers_params,
                                                   top_layer_params,
                                                   model_seed,
                                                   num_labels,
                                                   image_size))
            Connector.connect(self.inputs.label.output,
                              self._group.inputs.label)

            self._top_layer = self._group.top_layer
        self._conv_layers = self._group.conv_layers

        Connector.connect(self.inputs.image.output,
                          self._group.inputs.image)

    def _get_flock(self, layer_id: int) -> Union[ConvExpertFlockNode, SpatialPoolerFlockNode]:

        if layer_id < len(self._conv_layers):
            assert len(self._conv_layers[layer_id].expert_flock_nodes) == 1, \
                f'only num_flocks=1 supported here'
            return self._conv_layers[layer_id].expert_flock_nodes[0]

        elif layer_id == len(self._conv_layers):
            assert self._top_layer is not None, \
                f'tried to access the top_layer externally, but the top_layer not initialized!'
            return self._top_layer.sp_node

        raise ValueError(f"requested layer {layer_id} out of range of layers conv:{len(self._conv_layers)} + 1")

    def get_average_boosting_duration_for(self, layer_id: int) -> float:
        return average_boosting_duration(
            FlockNodeAccessor.get_sp_boosting_durations(self._get_flock(layer_id)))

    def get_num_boosted_clusters_ratio(self, layer_id: int) -> float:
        return num_boosted_clusters(
            FlockNodeAccessor.get_sp_boosting_durations(self._get_flock(layer_id)))

    def get_flock_size_of(self, layer_id: int) -> int:
        return self._get_flock(layer_id).params.flock_size

    def get_sp_size_for(self, layer_id: int) -> int:
        params = self._get_flock(layer_id).params
        return params.n_cluster_centers * params.flock_size

    def get_output_id_for(self, layer_id: int) -> int:
        return FlockNodeAccessor.get_sp_output_id(self._get_flock(layer_id))

    def clone_sp_output_tensor_for(self, layer_id: int) -> torch.Tensor:
        flock = self._get_flock(layer_id)
        flock_size = self.get_flock_size_of(layer_id)

        if type(flock) is ExpertFlockNode or type(flock) is ConvExpertFlockNode:
            # turns out that the conv expert can have shape [2,2,num_cc]
            return FlockNodeAccessor.get_sp_output_tensor(flock).clone().view(flock_size, -1)
        if type(flock) is SpatialPoolerFlockNode or type(flock) is ConvSpatialPoolerFlockNode:
            return SpatialPoolerFlockNodeAccessor.get_output_tensor(flock).clone().view(flock_size, -1)

        logger.error(f'unsupported flock class type')

    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        return self._top_layer.outputs.label.tensor.clone()

    def get_average_log_delta_for(self, layer_id: int) -> float:
        """Supports getting deltas just for the layer 0"""
        expert = self._get_flock(layer_id)

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

    # train/test
    def model_switch_to_training(self):
        self._switch_training(True)

    def model_switch_to_testing(self):
        self._switch_training(False)

    def _switch_training(self, training: bool):
        logger.info(f'Topology: changing the learning state to learning_on: {training}')
        try:
            for layer in self._conv_layers:
                layer.switch_learning(training)
            self._top_layer.switch_learning(training)

            self._is_learning = training
            logger.info(f'switching learning to {training} ')
        except:
            logger.info(f'unable to switch the learning to {training}')

    def is_learning(self):
        return self._is_learning

    def num_layers(self):
        return len(self._conv_layers) + 1

