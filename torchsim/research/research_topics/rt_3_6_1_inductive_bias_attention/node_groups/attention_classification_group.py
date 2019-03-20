import logging

import torch
import math

from torchsim.core.eval.metrics.sp_convergence_metrics import average_sp_delta
from torchsim.core.eval.node_accessors.sp_node_accessor import SpatialPoolerFlockNodeAccessor
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_base import NodeBase
from torchsim.core.graph.slots import InputSlot, OutputSlotBase
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import SpatialPoolerFlockNode, ExpertFlockNode, UnsqueezeNode
from torchsim.core.nodes.bottom_up_attention_group import BottomUpAttentionGroup
from torchsim.research.research_topics.rt_2_1_2_learning_rate2.node_groups.classification_model_group import \
    ClassificationModelGroup
from torchsim.significant_nodes import SpReconstructionLayer

logger = logging.getLogger(__name__)


class AttentionClassificationGroup(ClassificationModelGroup):
    def __init__(self,
                 use_attention: bool,
                 num_labels: int = 20,
                 input_data_size: int = 24 * 24 * 3,
                 n_cluster_centers: int = 20,
                 use_middle_layer: bool = False,
                 use_temporal_pooler: bool = True,
                 n_middle_layer_cluster_centers: int = 20):
        super().__init__("Attention Classification Group")

        self._last_output = self.inputs.image.output
        reconstruction_input_size = input_data_size
        if use_attention:
            self._add_layer(*self._make_attention_layer())
        if use_middle_layer:
            self._add_middle_layer(use_temporal_pooler, n_middle_layer_cluster_centers)
            reconstruction_input_size = n_middle_layer_cluster_centers
        self._top_layer: SpReconstructionLayer \
            = self._add_layer(*self._make_top_layer(num_labels, reconstruction_input_size, n_cluster_centers))

        Connector.connect(self.inputs.label.output, self._top_layer.inputs.label)

    def _add_layer(self,
                   layer: NodeBase,
                   input_slot: InputSlot,
                   output_slot: OutputSlotBase) -> NodeBase:
        self.add_node(layer)
        if self._last_output is not None:
            Connector.connect(self._last_output, input_slot)
        self._last_output = output_slot
        return layer

    @staticmethod
    def _make_attention_layer() -> (BottomUpAttentionGroup, InputSlot, OutputSlotBase):
        group = BottomUpAttentionGroup()
        return group, group.inputs.image, group.outputs.fof

    def _add_middle_layer(self,
                          use_temporal_pooler: bool,
                          n_middle_layer_cluster_centers: int):
        unsqueeze_node = UnsqueezeNode(dim=0)
        self._add_layer(unsqueeze_node, unsqueeze_node.inputs.input, unsqueeze_node.outputs.output)
        self._add_layer(*self._make_middle_layer_expert(use_temporal_pooler, n_middle_layer_cluster_centers))

    @staticmethod
    def _make_middle_layer_expert(use_temporal_pooler: bool, n_middle_layer_cluster_centers: int) \
            -> (NodeBase, InputSlot, OutputSlotBase):
        params = ExpertParams()
        params.flock_size = 1
        params.n_cluster_centers = n_middle_layer_cluster_centers
        if use_temporal_pooler:
            expert_node: ExpertFlockNode = ExpertFlockNode(params)
            input_slot = expert_node.inputs.sp.data_input
            output_slot = expert_node.outputs.tp.projection_outputs
            node = expert_node
        else:
            sp_node: SpatialPoolerFlockNode = SpatialPoolerFlockNode(params)
            input_slot = sp_node.inputs.sp.data_input
            output_slot = sp_node.outputs.sp.forward_clusters
            node = sp_node
        return node, input_slot, output_slot

    @staticmethod
    def _make_top_layer(num_labels: int,
                        input_data_size: int,
                        n_cluster_centers: int) -> (SpReconstructionLayer, InputSlot, OutputSlotBase):
        sp_params = ExpertParams()
        sp_params.flock_size = 1
        sp_params.n_cluster_centers = n_cluster_centers
        sp_params.compute_reconstruction = True
        layer = SpReconstructionLayer(input_data_size=input_data_size,
                                      labels_size=num_labels,
                                      sp_params=sp_params,
                                      name='TOP')
        return layer, layer.inputs.data, None

    def get_average_log_delta_for(self, layer_id: int) -> float:
        expert = self._top_layer.sp_node
        deltas = SpatialPoolerFlockNodeAccessor.get_sp_deltas(expert)
        delta = average_sp_delta(deltas)

        if delta > 0:  # avoid math.domain error
            delta = math.log(delta)
        return delta

    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        return self._top_layer.outputs.label.tensor.clone()

    def model_switch_to_training(self):
        self._switch_training(True)

    def model_switch_to_testing(self):
        self._switch_training(False)

    def _switch_training(self, training: bool):
        logger.info(f'Topology: changing the learning state to learning_on: {training}')
        self._top_layer.switch_learning(training)
        self._is_learning = training

