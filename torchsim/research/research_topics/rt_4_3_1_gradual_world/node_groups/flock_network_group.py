from typing import Optional

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.core.nodes.flock_networks.neural_network_flock import NeuralNetworkFlockParams, OutputActivation
from torchsim.core.nodes.network_flock_node import NetworkFlockNodeParams, NetworkFlockNode
from torchsim.research.research_topics.rt_4_3_1_gradual_world.topologies.general_gl_topology import PredictorGroupInputs, \
    PredictorGroupOutputs


class FlockNetworkGroup(NodeGroupBase[PredictorGroupInputs, PredictorGroupOutputs]):

    def __init__(self,
                 num_predictors: int,
                 learning_rate: Optional[float] = 0.1,
                 coefficients_minimum_max: float = 0.1,
                 hidden_size: int = 10,
                 n_layers: int = 1,
                 output_activation: Optional[OutputActivation] = OutputActivation.IDENTITY,
                 name: str = "FlockNetworkGroup",
                 ):

        super().__init__(name, inputs=PredictorGroupInputs(self), outputs=PredictorGroupOutputs(self))

        p_node_params = NetworkFlockNodeParams()
        p_node_params.flock_size = num_predictors
        p_node_params.do_delay_coefficients = False
        p_node_params.do_delay_input = True
        p_node_params.normalize_coefficients = False
        p_node_params.negative_coefficients_removal = True
        p_node_params.buffer_size = 500
        p_node_params.batch_size = 400
        p_node_params.learning_period = 20

        p_network_params = NeuralNetworkFlockParams()
        p_network_params.flock_size = p_node_params.flock_size
        p_network_params.input_size = 1  # determined form the input size in the Node._derive_params
        p_network_params.hidden_size = hidden_size
        p_network_params.output_size = 1  # determined from the  target size in the Node._derive_params
        p_network_params.output_activation = output_activation
        p_network_params.learning_rate = learning_rate
        p_network_params.coefficients_minimum_max = coefficients_minimum_max
        p_network_params.n_hidden_layers = n_layers

        predictors = NetworkFlockNode(node_params=p_node_params, network_params=p_network_params, name="Predictors")
        self.predictors = predictors
        self.add_node(predictors)

        # input of the group to both input and target of the flock
        Connector.connect(self.inputs.data.output, predictors.inputs.input_data)
        Connector.connect(self.inputs.data.output, predictors.inputs.target_data)

        # learning coefficients
        Connector.connect(self.inputs.learning_coefficients.output, predictors.inputs.learning_coefficients)

        # flock -> group outputs
        Connector.connect(predictors.outputs.prediction_output, self.outputs.predictors_activations.input)
        Connector.connect(predictors.outputs.error_output, self.outputs.predictors_activation_errors.input)

    def switch_learning(self, on):
        self.predictors.is_learning_enabled = on

