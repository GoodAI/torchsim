import random

import pytest
import torch

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.nn_node import NNetParams, NNetNode
from torchsim.core.utils.tensor_utils import same
from torchsim.topologies.nnet_topology import NNet
from torchsim.utils.baselines_utils import ObservationStorage
import torch.optim as optim
from torchsim.utils.seed_utils import set_global_seeds
from torchsim.utils.template_utils.train_test_topology_saver import PersistableSaver


def test_params_equality_implementation():
    """Test the __eq__ method implementation in the params"""

    input_size = 11
    num_classes = 123
    seed = None

    _params = NNetParams(NNetParams.default_params())
    _params.input_shape = (3, input_size, input_size)
    _params.output_size = num_classes
    _params.seed = seed

    assert _params == _params

    _params2 = NNetParams(NNetParams.default_params())
    _params2.input_shape = (3, input_size, input_size)
    _params2.output_size = num_classes
    _params2.seed = seed

    assert _params == _params2
    _params.seed = 1
    assert _params != _params2


def test_observation_storage_equality_implementation():
    """Storages should be equal until one network makes at least one different step"""

    _, net1 = make_nnet()
    _, net2 = make_nnet()

    input_tensor, label_tensor, _ = create_network_inputs(net1, net2)
    assert net1._storage == net2._storage  # initialization

    run_num_steps(3, input_tensor, label_tensor, net1, net2)
    assert net1._storage == net2._storage  # same num steps, same data

    run_num_steps(1, input_tensor, label_tensor, net1)
    assert net1._storage != net2._storage  # different num steps

    run_num_steps(1, input_tensor, label_tensor, net2)
    assert net1._storage == net2._storage  # same num steps, same data


def are_layers_same(layer1, layer2) -> bool:
    """ Compare layers of the two networks"""

    if type(layer1) != type(layer2):
        return False
    if not same(layer1.bias.data, layer2.bias.data):
        return False
    if not same(layer1.weight, layer2.weight):
        return False

    if layer1.training != layer2.training:
        return False

    return True


def are_nnet_modules_same(nn1: NNet, nn2: NNet) -> bool:
    """Compare actual torch.nn.Modules inside the NNetNode (unable to implement the __eq__ method inside the module)"""

    if not are_layers_same(nn1.conv1, nn2.conv1):
        return False

    if not are_layers_same(nn1.conv2, nn2.conv2):
        return False

    if not are_layers_same(nn1.fc1, nn2.fc1):
        return False

    if not are_layers_same(nn1.fc2, nn2.fc2):
        return False

    if not are_layers_same(nn1.fc3, nn2.fc3):
        return False

    return True


def are_complete_nodes_identical(node1: NNetNode, node2: NNetNode) -> bool:
    """Tests wheter the complete NNetNodes are in identical states"""

    # check most of the internal states
    if node1._params != node2._params:
        return False
    if node1._storage != node2._storage:
        return False
    if not are_nnet_modules_same(node1._network, node2._network):
        return False

    # check outputs
    if not same(node1.outputs.label.tensor, node2.outputs.label.tensor):
        return False
    if not same(node1.outputs.output.tensor, node2.outputs.output.tensor):
        return False
    if not same(node1.outputs.task_control.tensor, node2.outputs.task_control.tensor):
        return False

    return True


def test_nnet_module_equality():
    """Compare weights and biases of each layer"""

    _, net1 = make_nnet()
    _, net2 = make_nnet()

    nn1 = net1._network
    nn2 = net2._network

    input_tensor, label_tensor, _ = create_network_inputs(net1, net2)
    assert are_nnet_modules_same(nn1, nn2)  # after init

    run_num_steps(10, input_tensor, label_tensor, net1, net2)
    assert are_nnet_modules_same(nn1, nn2)  # after several steps without learning

    run_num_steps(1, input_tensor, label_tensor, net1)
    assert are_nnet_modules_same(nn1, nn2)  # one networks makes further steps

    # TODO learning is nondeterministic (nondeterministic sampling from the buffer)


def make_nnet(input_size: int = 16,
              num_classes: int = 5,
              seed: int = 123,
              buffer_s: int = 16,
              batch_s: int = 8,
              mixed_mode: bool = True):
    device = 'cuda'

    # set topology params and configs
    _params = NNetParams(NNetParams.default_params())
    # _params.set_params(_nn_node_params)  # params defined in this file
    # ._params.set_params(kwargs)  # params defined in the constructor

    # small input sizes for testing
    _params.input_shape = (3, input_size, input_size)
    _params.output_size = num_classes
    _params.seed = seed
    _params.batch_size = batch_s
    _params.buffer_size = buffer_s
    _params.num_epochs = 3

    _params.mixed_mode = mixed_mode

    # observation storage params
    _observation_types = {
        'x': (_params.buffer_size, *_params.input_shape),  # observations
        'y': (_params.buffer_size, _params.output_size),  # labels
    }

    # data storage
    _storage = ObservationStorage(
        _params.buffer_size,
        _observation_types)
    _storage.to('cpu' if _params.mixed_mode else device)

    # network needs to have the global seeds to have set before creating (outside of the node in this case)
    set_global_seeds(seed=_params.seed)

    # neural network setup
    _network = NNet(
        input_shape=_params.input_shape,
        output_shape=_params.output_size
    ).to('cuda' if _params.mixed_mode else device)

    # neural net optimizer
    _optimizer = optim.Adam(_network.parameters(), lr=_params.lr)

    # NNet Node
    _nnet_node = NNetNode(
        _network,
        _optimizer,
        _storage,
        _params,
        name='Neural Network Node')

    creator = AllocatingCreator(device=device)
    _nnet_node.allocate_memory_blocks(creator)

    return _params, _nnet_node


def _random_image(input_shape) -> torch.Tensor:
    return torch.randn(input_shape).permute(1, 2, 0)


def _random_label(num_classes: int) -> torch.Tensor:
    result = torch.zeros(num_classes)
    result[random.randint(0, num_classes - 1)] = 1
    return result


def create_network_inputs(net1: NNetNode, net2: NNetNode):
    """Create two identical networks, connect to the same inputs"""

    # generate input memory blocks
    input_tensor = _random_image(net1._params.input_shape)
    input_memory_block = MemoryBlock('input')
    input_memory_block.tensor = input_tensor

    input_label = _random_label(net1._params.output_size)
    label_memory_block = MemoryBlock('input_label')
    label_memory_block.tensor = input_label

    is_testing_tensor = torch.tensor([0])
    is_testing_memory_block = MemoryBlock('is_testing')
    is_testing_memory_block.tensor = is_testing_tensor

    # connect inputs to the network
    Connector.connect(input_memory_block, net1.inputs.input)
    Connector.connect(label_memory_block, net1.inputs.label)
    Connector.connect(is_testing_memory_block, net1.inputs.testing_phase)

    Connector.connect(input_memory_block, net2.inputs.input)
    Connector.connect(label_memory_block, net2.inputs.label)
    Connector.connect(is_testing_memory_block, net2.inputs.testing_phase)

    return input_tensor, input_label, is_testing_tensor


def run_num_steps(num_steps: int,
                  input_tensor: torch.Tensor,
                  input_label: torch.Tensor,
                  nnet_1: NNetNode,
                  nnet_2: NNetNode = None):

    for nnet in [nnet_1, nnet_2]:

        set_global_seeds(123)

        if nnet is not None:
            for step in range(num_steps):
                # print(f'----------------------------- step {step}')
                input_tensor.copy_(_random_image(nnet_1._params.input_shape))
                input_label.copy_(_random_label(nnet_1._params.output_size))

                # make the step with all networks we have
                nnet.step()


class NNetMockTopology(Topology):

    my_net: NNetNode

    def __init__(self, device: str):
        super().__init__(device)
        self.my_net = None

    def add(self, nnet):
        self.my_net = nnet
        self.add_node(nnet)


def test_net_deterministic_inference():
    """Two identical networks should be identical and provide identical outputs"""

    input_size = 31
    num_classes = 5

    buffer_s = 10
    batch_s = 5
    mixed_mode = True

    _params, net1 = make_nnet(input_size, num_classes, batch_s=batch_s, buffer_s=buffer_s, mixed_mode=mixed_mode)
    _, net2 = make_nnet(input_size, num_classes, batch_s=batch_s, buffer_s=buffer_s, mixed_mode=mixed_mode)

    input_tensor, input_label, is_testing_tensor = create_network_inputs(net1, net2)
    assert are_complete_nodes_identical(net1, net2)  # after initialization

    run_num_steps(5, input_tensor, input_label, net1, net2)
    assert are_complete_nodes_identical(net1, net2)  # after several steps before first learning

    run_num_steps(30, input_tensor, input_label, net1, net2)

    assert are_complete_nodes_identical(net1, net2)  # after first learning

    run_num_steps(10, input_tensor, input_label, net1)
    assert not are_complete_nodes_identical(net1, net2)


def test_net_serialization():
    """Test that the NNetNode can be serialized and deserialized and result is correct"""
    # TODO this identifies the potential problems with the serialization/deserialization
    # TODO these are not solved yet and for the nnet experiments, the deserialization should be disabled (hack)

    input_size = 31
    num_classes = 5

    buffer_s = 10
    batch_s = 5
    mixed_mode = True

    # make two networks, the first one will be saved/loaded, the second one not
    _params, net1 = make_nnet(input_size, num_classes, batch_s=batch_s, buffer_s=buffer_s, mixed_mode=mixed_mode)
    _, net2 = make_nnet(input_size, num_classes, batch_s=batch_s, buffer_s=buffer_s, mixed_mode=mixed_mode)

    input_tensor, input_label, is_testing_tensor = create_network_inputs(net1, net2)

    # save/load infrastructure..
    saver = PersistableSaver('saved_nnet_test')
    topology = NNetMockTopology(device='cuda')
    topology.add(net1)
    topology.prepare()

    assert are_complete_nodes_identical(topology.my_net, net2)  # after initialization

    saver.save_data_of(topology)
    assert are_complete_nodes_identical(topology.my_net, net2)  # after saving..

    saver.load_data_into(topology)
    assert are_complete_nodes_identical(topology.my_net, net2)  # after loading..

    run_num_steps(5, input_tensor, input_label, topology.my_net, net2)
    assert are_complete_nodes_identical(topology.my_net, net2)  # after several steps

    run_num_steps(30, input_tensor, input_label, topology.my_net)  # make more steps so that the topology.my_net changes weights
    assert not are_complete_nodes_identical(topology.my_net, net2)  # my_net should be different than net2

    assert are_nnet_modules_same(topology.my_net._network, net2._network)

    saver.load_data_into(topology)

    assert are_nnet_modules_same(topology.my_net._network, net2._network)

    # TODO deserialization not implemented for the sampler
    # assert are_complete_nodes_identical(topology.my_net, net2)  # load back original state of the topology.my_net










































