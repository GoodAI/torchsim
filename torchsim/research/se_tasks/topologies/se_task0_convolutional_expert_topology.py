import torch

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import ConvExpertFlockNode, ExpertFlockNode, LambdaNode, ReceptiveFieldNode, \
    SpatialPoolerFlockNode, UnsqueezeNode
from torchsim.gui.observables import ObserverPropertiesItem
from torchsim.gui.ui_utils import parse_bool
from torchsim.research.se_tasks.topologies.se_task0_topology import SeT0TopologicalGraph
from torchsim.utils.param_utils import Size2D


class SeT0ConvTopology(SeT0TopologicalGraph):
    EXPERTS_IN_X = 4
    BATCH = 2000
    TRAINING = True

    parent_params = ExpertParams()
    parent_params.flock_size = 1
    parent_params.n_cluster_centers = 80
    parent_params.compute_reconstruction = True
    parent_params.spatial.learning_period = 400
    parent_params.spatial.learning_rate = 0.5
    if not TRAINING:
        parent_params.spatial.learning_rate *= 0

    parent_params.spatial.buffer_size = BATCH
    parent_params.spatial.batch_size = BATCH
    parent_params.spatial.cluster_boost_threshold = 200

    mid_expert_params = ExpertParams()
    mid_expert_params.flock_size = 1
    mid_expert_params.n_cluster_centers = 100
    mid_expert_params.compute_reconstruction = True
    mid_expert_params.spatial.cluster_boost_threshold = 200
    mid_expert_params.spatial.learning_rate = 0.5
    if not TRAINING:
        mid_expert_params.spatial.learning_rate *= 0
    mid_expert_params.spatial.batch_size = BATCH
    mid_expert_params.spatial.buffer_size = BATCH
    mid_expert_params.spatial.learning_period = 400

    conv_flock_params = ExpertParams()
    conv_flock_params.flock_size = EXPERTS_IN_X ** 2
    conv_flock_params.n_cluster_centers = 600
    conv_flock_params.spatial.learning_period = 1000
    conv_flock_params.spatial.learning_rate = 0.2
    if not TRAINING:
        conv_flock_params.spatial.learning_rate *= 0
    conv_flock_params.spatial.buffer_size = BATCH
    conv_flock_params.spatial.batch_size = BATCH
    conv_flock_params.spatial.cluster_boost_threshold = 1000

    # parent_rf_stride_dims = (16, 16)

    def _install_experts(self):
        im_width = self.se_io.get_image_width()
        im_height = self.se_io.get_image_height()
        self.input_dims = torch.Size((im_height, im_width, 3))
        self.parent_rf_dims = Size2D(im_height // self.EXPERTS_IN_X, im_width // self.EXPERTS_IN_X)

        lrf_node = ReceptiveFieldNode(input_dims=self.input_dims, parent_rf_dims=self.parent_rf_dims)

        self.add_node(lrf_node)

        self._top_level_flock_node = SpatialPoolerFlockNode(self.parent_params)
        self._mid_node = ExpertFlockNode(self.mid_expert_params)
        self._conv_node = ConvExpertFlockNode(self.conv_flock_params)

        self.add_node(self._top_level_flock_node)
        self.add_node(self._mid_node)
        self.add_node(self._conv_node)

        def rescale(inputs, outputs):
            if self.TRAINING:
                outputs[0].copy_(inputs[0] * 1000)  # large constant to make the label more important
            else:
                outputs[0].copy_(inputs[0] * float('nan'))

        self._rescale_node = LambdaNode(rescale, 1, [(self.se_io.get_num_labels(),)])
        self.add_node(self._rescale_node)

        unsqueeze_node_0 = UnsqueezeNode(0)
        self.add_node(unsqueeze_node_0)
        unsqueeze_node_1 = UnsqueezeNode(0)
        self.add_node(unsqueeze_node_1)

        Connector.connect(self.se_io.outputs.image_output, lrf_node.inputs.input)

        Connector.connect(lrf_node.outputs.output, self._conv_node.inputs.sp.data_input)
        Connector.connect(self._conv_node.outputs.tp.projection_outputs, unsqueeze_node_0.inputs.input)
        Connector.connect(unsqueeze_node_0.outputs.output, self._mid_node.inputs.sp.data_input)
        Connector.connect(self._mid_node.outputs.tp.projection_outputs, self._join_node.inputs[0])

        Connector.connect(self.se_io.outputs.task_to_agent_label, self._rescale_node.inputs[0])
        Connector.connect(self._rescale_node.outputs[0], self._join_node.inputs[1])
        Connector.connect(self._join_node.outputs.output, unsqueeze_node_1.inputs.input)
        Connector.connect(unsqueeze_node_1.outputs.output, self._top_level_flock_node.inputs.sp.data_input)

    def _top_level_expert_output_size(self):
        return self.mid_expert_params.n_cluster_centers * self.mid_expert_params.flock_size

    def switch_training_testing(self, training: bool):
        if self._rescale_node._unit is None:
            return

        def rescale(inputs, outputs):
            if training:
                outputs[0].copy_(inputs[0] * 1000)  # large constant to make the label more important
            else:
                outputs[0].copy_(inputs[0] * float('nan'))

        self._rescale_node.change_function(rescale)

    _topology_test_checked: bool = False

    def get_properties(self):
        prop_list = super().get_properties()

        def topology_test_checked(value):
            self._topology_test_checked = value
            self.switch_training_testing(not parse_bool(value))
            return value

        prop_list.append(ObserverPropertiesItem("Test topology", 'checkbox', self._topology_test_checked,
                                                topology_test_checked))
        return prop_list

    def _get_agent_output(self) -> MemoryBlock:
        return self._top_level_flock_node.outputs.sp.current_reconstructed_input
