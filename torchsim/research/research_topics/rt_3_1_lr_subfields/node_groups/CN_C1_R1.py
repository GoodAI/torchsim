from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.core.models.expert_params import ExpertParams
from torchsim.significant_nodes.conv_layer import ConvLayerParams, create_connected_conv_layers, ConvLayer, SpConvLayer
from torchsim.significant_nodes.reconstruction_interface import ClassificationOutputs, ClassificationInputs
from torchsim.significant_nodes.sp_reconstruction_layer import SpReconstructionLayer


class CN_C1_R1(NodeGroupBase[ClassificationInputs, ClassificationOutputs]):
    conv_layer_class = ConvLayer

    def __init__(self, name: str = None, bottom_layer_size=5, l_0_cluster_centers=10,
                 l_1_cluster_centers=20, l_0_rf_dims=(3, 3), l_0_rf_stride=None, l_1_rf_dims=(2, 2), env_size=(24, 24),
                 label_length=3, sp_n_cluster_centers=10, l_0_custom_args=None):
        if name is None:
            if hasattr(ConvLayer, 'SHORT_NAME'):
                name = f"{ConvLayer.SHORT_NAME}{bottom_layer_size}_{ConvLayer.SHORT_NAME}1_R1"
            else:
                name = f"?{bottom_layer_size}_?1_R1"

        super().__init__(name, inputs=ClassificationInputs(self), outputs=ClassificationOutputs(self))

        ep_l_0 = ExpertParams()
        ep_l_0.spatial.learning_rate = 0.1
        ep_l_0.spatial.cluster_boost_threshold = 1000
        ep_l_0.spatial.max_boost_time = 2000
        ep_l_0.spatial.buffer_size = 225 * 200
        ep_l_0.spatial.learning_period = 300
        ep_l_0.temporal.n_frequent_seqs = 2000
        ep_l_0.temporal.max_encountered_seqs = 4000

        ep_l_1 = ExpertParams()
        ep_l_1.spatial.learning_rate = 0.1
        ep_l_1.spatial.cluster_boost_threshold = 1000
        ep_l_1.spatial.max_boost_time = 2000
        ep_l_1.spatial.buffer_size = 225 * 200
        ep_l_1.spatial.learning_period = 300
        ep_l_1.temporal.n_frequent_seqs = 2000
        ep_l_1.temporal.max_encountered_seqs = 4000

        ep_l_0.n_cluster_centers = l_0_cluster_centers
        ep_l_1.n_cluster_centers = l_1_cluster_centers

        bottom_layer_params = ConvLayerParams(ep_l_0, l_0_rf_dims, "L0", conv_layer_class=self.conv_layer_class,
                                              num_flocks=bottom_layer_size, input_rf_stride=l_0_rf_stride,
                                              custom_args=l_0_custom_args)
        l1_params = ConvLayerParams(ep_l_1, l_1_rf_dims, "L1", conv_layer_class=self.conv_layer_class)

        layers_params = [bottom_layer_params, l1_params]

        if len(env_size) == 2:
            env_size += (1,)
        self.conv_layers, output_projection_size = \
            create_connected_conv_layers(layers_params, env_size)

        for layer in self.conv_layers:
            self.add_node(layer)

        conv_layer_0 = self.conv_layers[0]
        conv_layer_1 = self.conv_layers[-1]

        # topmost layer
        ep_sp = ExpertParams()
        ep_sp.flock_size = 1
        ep_sp.n_cluster_centers = sp_n_cluster_centers
        sp_reconstruction_layer = SpReconstructionLayer(output_projection_size,
                                                        label_length,
                                                        sp_params=ep_sp, name="L2")
        self.add_node(sp_reconstruction_layer)
        self.sp_reconstruction_layer = sp_reconstruction_layer

        Connector.connect(self.inputs.data.output, conv_layer_0.inputs.data)
        Connector.connect(conv_layer_1.outputs.data, sp_reconstruction_layer.inputs.data)

        Connector.connect(self.inputs.label.output, sp_reconstruction_layer.inputs.label)

        self.is_training = True

    def switch_tt(self, train: bool):
        self.is_training = train

        for conv_layer in self.conv_layers:
            conv_layer.switch_learning(train)
        self.sp_reconstruction_layer.switch_learning(train)

    def restart(self):
        pass


class SCN_SC1_R1(CN_C1_R1):
    conv_layer_class = SpConvLayer
