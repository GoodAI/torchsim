from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.core.nodes import ConstantNode
from torchsim.core.nodes.flock_networks.neural_network_flock import NeuralNetworkFlockParams, OutputActivation
from torchsim.core.nodes.network_flock_node import NetworkFlockNodeParams, NetworkFlockNode
from torchsim.core.nodes.squeeze_node import SqueezeNode
from torchsim.research.research_topics.rt_4_3_1_gradual_world.topologies.general_gl_topology import GateGroupInputs,\
    GateGroupOutputs


class GateNetworkGroup(NodeGroupBase[GateGroupInputs, GateGroupOutputs]):

    def __init__(self,
                 num_predictors: int,
                 learning_rate: float = 0.05,
                 hidden_size: int = 10,
                 name: str = "GateNetworkGroup"):

        super().__init__(name,inputs=GateGroupInputs(self), outputs=GateGroupOutputs(self))

        # gate
        g_node_params = NetworkFlockNodeParams()
        g_node_params.flock_size = 1
        g_node_params.do_delay_coefficients = False
        g_node_params.do_delay_input = True
        g_node_params.learning_period = 20
        g_node_params.buffer_size = 1000
        g_node_params.batch_size = 900

        g_network_params = NeuralNetworkFlockParams()
        g_network_params.input_size = 1  # this should be determined automatically from the input shape
        g_network_params.mini_batch_size = 100
        g_network_params.hidden_size = hidden_size
        g_network_params.output_size = num_predictors  # might be determined form the input target size
        g_network_params.output_activation = OutputActivation.SOFTMAX
        g_network_params.learning_rate = learning_rate

        # gate itself
        self.gate = NetworkFlockNode(node_params=g_node_params, network_params=g_network_params, name="Gate Network")
        self.add_node(self.gate)

        # const
        learning_constant = ConstantNode([1], 1)
        self.add_node(learning_constant)

        # squeeze
        squeeze_node = SqueezeNode(0)
        self.add_node(squeeze_node)

        Connector.connect(learning_constant.outputs.output, self.gate.inputs.learning_coefficients)
        Connector.connect(self.inputs.data.output, self.gate.inputs.input_data)
        Connector.connect(self.inputs.targets.output, self.gate.inputs.target_data)

        Connector.connect(self.gate.outputs.prediction_output, squeeze_node.inputs[0])
        Connector.connect(squeeze_node.outputs[0], self.outputs.outputs.input)

    def switch_learning(self, on):
        self.gate.is_learning_enabled = on
