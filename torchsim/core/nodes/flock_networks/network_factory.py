from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.nodes.flock_networks.multi_layer_perceptron_flock import MultilayerPerceptronFlock
from torchsim.core.nodes.flock_networks.neural_network_flock import NeuralNetworkFlockParams, NeuralNetworkFlockTypes, \
    NeuralNetworkFlock


def create_networks(network_params: NeuralNetworkFlockParams,
                    creator: TensorCreator,
                    network_type: NeuralNetworkFlockTypes) -> NeuralNetworkFlock:
    """Create a network of selected type"""

    if network_type == NeuralNetworkFlockTypes.MLP:
        return MultilayerPerceptronFlock(network_params, creator.device)
    else:
        raise ValueError("Unknown NeuralNetworkFlockType")
