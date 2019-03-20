from typing import Tuple

import pytest

import torch
from torchsim.core import get_float
from torchsim.core.nodes.flock_networks.multi_layer_perceptron_flock import MultilayerPerceptronFlock, \
    NeuralNetworkFlockParams
from torchsim.core.utils.tensor_utils import same


def prepare_inputs_and_targets(flock_size, device: str) -> Tuple[int, int, int, torch.Tensor, torch.Tensor]:
    batch_inputs = torch.tensor([[0, .8, .2, 0], [.5, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, .1]], device=device)

    batch_size = batch_inputs.shape[0]
    input_size = batch_inputs.shape[1]
    flock_inputs = batch_inputs.unsqueeze(0).expand(flock_size, batch_size, input_size)

    batch_targets = torch.tensor([[-1, 1], [.9, -1], [-1, -.8], [1, 1]], device=device)
    output_size = batch_targets.shape[1]
    flock_targets = batch_targets.unsqueeze(0).expand(flock_size, batch_size, output_size)

    return input_size, output_size, batch_size, flock_inputs, flock_targets


def test_ta_neural_nets():
    """Tests set of NNs for use in NN flock node."""

    params = NeuralNetworkFlockParams()

    params.flock_size = 3
    params.mini_batch_size = 2

    params.hidden_size = 3
    params.learning_rate = .1
    device = 'cuda'

    params.input_size, params.output_size, batch_size, flock_inputs, flock_targets = \
        prepare_inputs_and_targets(params.flock_size, device)

    neural_nets = MultilayerPerceptronFlock(params, device)

    batch_learning_weights = torch.tensor([.1, .2, .4, 1], device=device)
    flock_learning_weights = batch_learning_weights.unsqueeze(0).unsqueeze(2).expand(params.flock_size, batch_size, 1)

    data_loaders, should_learn = neural_nets.make_data_loaders(flock_inputs,
                                                               flock_targets,
                                                               flock_learning_weights,
                                                               params.mini_batch_size,
                                                               params.coefficients_minimum_max)
    neural_nets.train(data_loaders, should_learn)
    errors = neural_nets.test(flock_inputs, flock_targets, device=device)
    assert errors.numel() == params.flock_size
    mean_error = errors.mean().item()
    assert 4 > mean_error >= 0


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize('learning_coefficients', [
    ([[0, 0.5, 1, 0.9], [1, 0, 0, 0.9], [0, 1, 0.2, 0.4]]),
    ([[0, 0, 0, 0.01]])
])
def test_coefficients(learning_coefficients):
    """Test that the data with learning_coefficients equal zero will not be sampled."""

    params = NeuralNetworkFlockParams()

    params.flock_size = len(learning_coefficients)
    params.mini_batch_size = 57

    device = 'cuda'

    params.input_size, params.output_size, batch_size, flock_inputs, flock_targets = \
        prepare_inputs_and_targets(params.flock_size, device)

    neural_nets = MultilayerPerceptronFlock(params, device)

    batch_learning_coefficients = torch.tensor(learning_coefficients, device=device, dtype=get_float(device))
    batch_learning_coefficients = batch_learning_coefficients.unsqueeze(2)

    data_loaders, should_learn = neural_nets.make_data_loaders(flock_inputs,
                                                               flock_targets,
                                                               batch_learning_coefficients,
                                                               params.mini_batch_size,
                                                               params.coefficients_minimum_max)
    # Compute frequency of each datapoint
    for flock_idx, loader in enumerate(data_loaders):
        for input_batch, target_batch in loader:
            occurrences = []
            for point_idx, point in enumerate(flock_inputs[flock_idx]):
                seen = 0
                for sampled_point in input_batch:
                    if same(sampled_point, point):
                        seen += 1

                occurrences.append(seen)
                # Data-points with 0 coefficients should not be present
                if learning_coefficients[flock_idx][point_idx] == 0:
                    assert seen == 0
                else:
                    assert seen > 0
