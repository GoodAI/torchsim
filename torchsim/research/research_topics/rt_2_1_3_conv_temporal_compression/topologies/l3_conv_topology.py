from typing import Tuple, Type, Union

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.significant_nodes import BallEnvironment, SpReconstructionLayer, BallEnvironmentParams, SEEnvironment, \
    SpConvLayer
from torchsim.significant_nodes.conv_layer import ConvLayerParams, create_connected_conv_layers, ConvLayer
from torchsim.significant_nodes.environment_base import EnvironmentParamsBase


class L3ConvTopology(Topology):
    """
    SPReconstructionLayer
           ^         ^
           |         |
       ConvLayer     |
           ^         |
           |         |
         n-times     |
           |         |
       ConvLayer     |
           ^         |
           |         |
          Env  -> Switch <- NaNs
    """

    conv_class = ConvLayer

    def __init__(self,
                 env_class: Union[Type[BallEnvironment], Type[SEEnvironment]]=BallEnvironment,
                 env_params: EnvironmentParamsBase=BallEnvironmentParams(),
                 l_0_rf_dims: Tuple[int, int] = (3, 3),
                 l_1_rf_dims: Tuple[int, int] = (2, 2),
                 sp_n_cluster_centers=20,
                 l_0_cluster_centers=50,
                 l_1_cluster_centers=50):
        super().__init__('cuda')
        # params
        # l_0_rf_dims depends on env_size (must divide without remainder)
        # l_1_rf_dims is dividing ep_l_0.flock_size (*(env_size//l_0_rf_dims)) without remainder

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

        bottom_layer_params = ConvLayerParams(ep_l_0, l_0_rf_dims, "L0", conv_layer_class=self.conv_class)
        l1_params = ConvLayerParams(ep_l_1, l_1_rf_dims, "L1", conv_layer_class=self.conv_class)

        layers_params = [bottom_layer_params, l1_params]

        # code
        # conv part
        env_node = env_class(env_params)
        self.add_node(env_node)
        self.env_node = env_node

        self.conv_layers, output_projection_size = \
            create_connected_conv_layers(layers_params, env_params.env_size)

        for layer in self.conv_layers:
            self.add_node(layer)

        conv_layer_0 = self.conv_layers[0]
        conv_layer_1 = self.conv_layers[-1]

        # topmost layer
        ep_sp = ExpertParams()
        ep_sp.flock_size = 1
        ep_sp.n_cluster_centers = sp_n_cluster_centers
        sp_reconstruction_layer = SpReconstructionLayer(output_projection_size,
                                                        env_params.n_shapes,
                                                        sp_params=ep_sp, name="L2")
        self.add_node(sp_reconstruction_layer)
        self.sp_reconstruction_layer = sp_reconstruction_layer

        Connector.connect(env_node.outputs.data, conv_layer_0.inputs.data)
        Connector.connect(conv_layer_1.outputs.data, sp_reconstruction_layer.inputs.data)

        Connector.connect(env_node.outputs.label, sp_reconstruction_layer.inputs.label)

        self.is_training = True

    def switch_tt(self, train: bool):
        self.is_training = train

        for conv_layer in self.conv_layers:
            conv_layer.switch_learning(train)
        self.sp_reconstruction_layer.switch_learning(train)
        self.env_node.switch_learning(train)

    def restart(self):
        # TODO: what should this method do?
        pass


class L3SpConvTopology(L3ConvTopology):
    """
    SPReconstructionLayer
           ^         ^
           |         |
      SpConvLayer    |
           ^         |
           |         |
      SpConvLayer    |
           ^         |
           |         |
        BallEnv -> Switch <- NaNs
    """

    conv_class = SpConvLayer
