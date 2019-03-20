from typing import List

import torch

from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import UnsqueezeNode, ConvSpatialPoolerFlockNode, LambdaNode, ReceptiveFieldNode, \
    SpatialPoolerFlockNode
from torchsim.research.se_tasks.topologies.se_task0_topology import SeT0TopologicalGraph
from torchsim.utils.param_utils import Size2D


class SeT0ConvSPTopology(SeT0TopologicalGraph):
    _flock_nodes: List[SpatialPoolerFlockNode]

    EXPERTS_IN_X = 8
    BATCH = 10000
    TRAINING = True

    parent_params = ExpertParams()
    parent_params.flock_size = 1
    parent_params.n_cluster_centers = 200
    parent_params.compute_reconstruction = True
    parent_params.spatial.learning_period = 400
    parent_params.spatial.learning_rate = 0.2
    if not TRAINING:
        parent_params.spatial.learning_rate *= 0

    parent_params.spatial.buffer_size = 1000
    parent_params.spatial.batch_size = 1000
    parent_params.spatial.cluster_boost_threshold = 200

    conv_flock_params = ExpertParams()
    conv_flock_params.flock_size = EXPERTS_IN_X ** 2
    conv_flock_params.n_cluster_centers = 200
    conv_flock_params.spatial.learning_period = 1000
    conv_flock_params.spatial.learning_rate = 0.2
    if not TRAINING:
        conv_flock_params.spatial.learning_rate *= 0
    conv_flock_params.spatial.buffer_size = BATCH
    conv_flock_params.spatial.batch_size = BATCH
    conv_flock_params.spatial.cluster_boost_threshold = 200

    input_dims = torch.Size((24, 24, 3))
    parent_rf_dims = Size2D(24 // EXPERTS_IN_X, 24 // EXPERTS_IN_X)

    # parent_rf_stride_dims = (16, 16)

    def _install_experts(self):
        lrf_node = ReceptiveFieldNode(input_dims=self.input_dims, parent_rf_dims=self.parent_rf_dims)

        self.add_node(lrf_node)

        self._top_level_flock_node = SpatialPoolerFlockNode(self.parent_params, name="Parent 1 SP")
        self._conv_node = ConvSpatialPoolerFlockNode(self.conv_flock_params, name="Conv SP flock")

        self.add_node(self._top_level_flock_node)
        self.add_node(self._conv_node)

        scale = 1000

        def rescale_up(inputs, outputs):
            outputs[0].copy_(inputs[0] * scale)

        def rescale_down(inputs, outputs):
            outputs[0].copy_(inputs[0] / scale)

        self._rescale_up_node = LambdaNode(rescale_up, 1, [(20,)], name="upscale 1000")
        self.add_node(self._rescale_up_node)

        self._rescale_down_node = LambdaNode(rescale_down, 1, [(1, self._top_level_expert_output_size() + 20)],
                                             name="downscale 1000")
        self.add_node(self._rescale_down_node)

        unsqueeze_node = UnsqueezeNode(0)
        self.add_node(unsqueeze_node)

        Connector.connect(self.se_io.outputs.image_output, lrf_node.inputs.input)

        Connector.connect(lrf_node.outputs.output, self._conv_node.inputs.sp.data_input)
        Connector.connect(self._conv_node.outputs.sp.forward_clusters, self._join_node.inputs[0])

        Connector.connect(self.se_io.outputs.task_to_agent_label, self._rescale_up_node.inputs[0])
        Connector.connect(self._rescale_up_node.outputs[0], self._join_node.inputs[1])
        Connector.connect(self._join_node.outputs.output, unsqueeze_node.inputs.input)
        Connector.connect(unsqueeze_node.outputs.output, self._top_level_flock_node.inputs.sp.data_input)

        Connector.connect(self._top_level_flock_node.outputs.sp.current_reconstructed_input,
                          self._rescale_down_node.inputs[0])

    def _get_agent_output(self):
        return self._rescale_down_node.outputs[0]

    def _top_level_expert_output_size(self):
        return self.conv_flock_params.n_cluster_centers * self.conv_flock_params.flock_size
