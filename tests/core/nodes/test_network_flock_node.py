from typing import List, Generator, Any, Tuple

import pytest
from torch.utils.data import DataLoader

import torch
import random

from torchsim.core import FLOAT_NAN
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.flock_networks.multi_layer_perceptron_flock import MultilayerPerceptronFlock
from torchsim.core.nodes.flock_networks.neural_network_flock import NeuralNetworkFlockParams, NeuralNetworkFlockTypes, \
    NeuralNetworkFlock
from torchsim.core.nodes.network_flock_node import create_delay_buffer, DelayBuffer, NetworkFlockNode, NetworkFlockNodeParams
from torchsim.core.utils.tensor_utils import same
from tests.core.nodes.node_unit_test_base import NodeTestBase, AnyResult
from itertools import product, repeat

from unittest.mock import patch


def _push(buffers: List[DelayBuffer],  data: torch.Tensor):
    for buffer in buffers:
        buffer.push(data)


def test_one_step_tensor_delay_buffer():
    """Test a buffer which delays the data one step"""
    shape = (3, 2, 1)

    creator = AllocatingCreator(device='cpu')

    buffer = create_delay_buffer(creator, True, shape)
    no_delay = create_delay_buffer(creator, False, shape)
    buffers = [buffer, no_delay]

    nans = creator.full(shape, FLOAT_NAN)

    a = creator.ones(shape)
    b = creator.ones(shape) * 2
    c = creator.ones(shape) * 3
    d = creator.ones(shape) * 4

    _push(buffers, a)
    assert same(buffer.read(), nans)
    assert same(no_delay.read(), a)
    _push(buffers, b)
    assert same(buffer.read(), a)
    assert same(no_delay.read(), b)
    _push(buffers, c)
    assert same(buffer.read(), b)
    assert same(no_delay.read(), c)
    _push(buffers, d)
    assert same(buffer.read(), c)
    assert same(no_delay.read(), d)


class TestNetworkFlockNodeChangingLrHasEffect(NodeTestBase):

    flock_size = 3

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        yield [
            # inputs
            self._creator.randn((2, 4), device=self._device, dtype=self._dtype),
            # targets
            self._creator.randn((5, 1, 2), device=self._device, dtype=self._dtype),
            # learning coefficients
            self._creator.randn((self.flock_size,), device=self._device, dtype=self._dtype)
        ]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        yield [
            AnyResult
        ]

    def _create_node(self) -> NetworkFlockNode:
        node_params = NetworkFlockNodeParams()
        node_params.flock_size = self.flock_size
        node_params.network_type = NeuralNetworkFlockTypes.MLP
        network_params = NeuralNetworkFlockParams()
        network_params.learning_rate = 0.3
        self.node = NetworkFlockNode(node_params=node_params, network_params=network_params)

        self.node.learning_rate = 0.2
        return self.node

    def _check_results(self, expected, results, step: int):
        super()._check_results(expected, results, step)

        # check also that the learning rate can be set
        networks: MultilayerPerceptronFlock = self.node._unit._networks
        for optimizer in networks.optimizers:
            for param_group in optimizer.param_groups:
                assert 0.2 == param_group['lr']


class DataCheckingDataLoader:
    """ Used to wrap a DataLoader, transparently checks data are correct when being iterated through. """
    def __init__(self, nn_id: int, loader: DataLoader):
        self.loader = loader
        self.nn_id = nn_id
        self.iter = None

    def __iter__(self):
        self.iter = iter(self.loader)
        return self

    def __next__(self):
        input_batch, target_batch = next(self.iter)
        assert same(input_batch, target_batch), 'All inputs should equal their targets'

        nns_targeted = torch.sum(input_batch > 0, dim=0)
        assert nns_targeted.argmax() == self.nn_id, 'The most examples should be for the targeted nn'

        return input_batch, target_batch


orig_make_data_loaders = NeuralNetworkFlock.make_data_loaders


def make_data_loaders(self: NeuralNetworkFlock, *args) -> Tuple[List[DataCheckingDataLoader], List[bool]]:
    train_loaders, should_learn = orig_make_data_loaders(self, *args)
    data_checking_train_loaders = []
    for i, loader in enumerate(train_loaders):
        data_checking_train_loaders.append(DataCheckingDataLoader(i, loader))
    return data_checking_train_loaders, should_learn


@patch('torchsim.core.nodes.flock_networks.neural_network_flock.NeuralNetworkFlock.make_data_loaders', new=make_data_loaders)
class TestNetworkFlockNodeDelaysCorrect(NodeTestBase):

    flock_size = 4
    steps = 35  # just 5 higher than batch_size => 30 steps collecting data + 5 learning steps (learning_period = 1)
    do_delay_input = False  # changed in test_node()
    do_delay_coefficients = False  # changed in test_node()

    def rand_id(self) -> int:
        return random.randrange(0, self.flock_size)

    def create_encoding(self, nn_id: int, seq_id: int) -> torch.Tensor:
        """ Produces a vector, which encodes both nn_id (as position in row) and seq_id (as placed value) in it. """
        t = self._creator.zeros((self.flock_size,), device=self._device, dtype=self._dtype)
        t[nn_id] = seq_id
        return t

    def create_probability(self, most_probable_id: int) -> torch.Tensor:
        """ Produces a vector containing a prob. distribution with highest prob goes to most_probable_id index. """
        max_other_val = 0.1
        t = self._creator.rand((self.flock_size,), device=self._device, dtype=self._dtype)
        t *= max_other_val
        t[most_probable_id] = 1
        t /= t.sum()
        return t

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:

        nn_id = self.rand_id()

        for seq_id in range(1, self.steps + 1):
            next_nn_id = self.rand_id()

            inputs = self.create_encoding(next_nn_id if self.do_delay_input else nn_id,
                                          seq_id + int(self.do_delay_input))
            targets = self.create_encoding(nn_id, seq_id)
            learning_coefficients = self.create_probability(next_nn_id if self.do_delay_coefficients else nn_id)

            nn_id = next_nn_id

            yield [
                inputs,
                targets,
                learning_coefficients
            ]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        yield from repeat([AnyResult], times=self.steps)

    def _create_node(self) -> NetworkFlockNode:
        node_params = NetworkFlockNodeParams()
        node_params.flock_size = self.flock_size
        node_params.network_type = NeuralNetworkFlockTypes.MLP
        node_params.do_delay_input = self.do_delay_input
        node_params.do_delay_coefficients = self.do_delay_coefficients

        network_params = NeuralNetworkFlockParams()
        network_params.learning_rate = 0.3
        network_params.mini_batch_size = 29
        node_params.learning_period = 1
        node_params.batch_size = 30
        self.node = NetworkFlockNode(node_params=node_params, network_params=network_params)

        return self.node

    @pytest.mark.flaky(reruns=3)
    def test_node(self):
        for delay_input, delay_coeffs in product((False, True), repeat=2):
            self.do_delay_input = delay_input
            self.do_delay_coefficients = delay_coeffs
            super().test_node()

    @pytest.mark.flaky(reruns=3)
    def test_serialization(self):
        super().test_serialization()




class TestNetworkFlockNodeDelaysCorrectCpu(TestNetworkFlockNodeDelaysCorrect):

    @classmethod
    def setup_class(cls, device: str = 'cuda'):
        device = 'cpu'
        super().setup_class(device)
