from typing import List

from torch.nn import Linear, ReLU, Module, Sequential
from torch.optim import Adam
from torch.utils.data import DataLoader

import torch
from torchsim.core.nodes.flock_networks.neural_network_flock import NeuralNetworkFlock, NeuralNetworkFlockParams, \
    OutputActivation
from torchsim.gui.validators import validate_predicate
from itertools import tee


def pairwise_consecutive(iterable):
    """"s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class MultiLayerPerceptron(Module):
    """Simple classifier with 0+ hidden layer(s) and softmax output."""

    def __init__(self, input_size: int, hidden_size: int, n_hidden_layers: int, output_size: int,
                 output_activation: OutputActivation, device: str):
        super().__init__()

        sizes = [input_size] + [hidden_size] * n_hidden_layers

        self._hidden = Sequential()
        for i, (in_size, out_size) in enumerate(pairwise_consecutive(sizes)):
            self._hidden.add_module(f'hidden_{i}', Linear(in_size, out_size))
            self._hidden.add_module(f'act_{i}', ReLU())

        self._out = Linear(sizes[-1], output_size)
        self._out_act = output_activation
        self._out_act_fn = output_activation.get_activation()
        self.to(device)

    def forward(self, input_data):
        hidden_activations = self._hidden(input_data)
        output_activations = self._out(hidden_activations)
        if self._out_act != OutputActivation.SOFTMAX or not self.training:
            output_activations = self._out_act_fn(output_activations)
        return output_activations


class MultilayerPerceptronFlock(NeuralNetworkFlock):
    """Set of neural networks used by neural network flock nodes."""

    _params: NeuralNetworkFlockParams

    def __init__(self,
                 params: NeuralNetworkFlockParams,
                 device: str):
        """Creates the neural networks.

        Args:
            params: params of the neural network(s)
            device: Device (cuda or cpu) used when creating tensors
        """

        self._params = params

        self.flock_size = params.flock_size
        self.input_size = params.input_size
        self.output_size = params.output_size
        self.nets = [MultiLayerPerceptron(params.input_size, params.hidden_size, params.n_hidden_layers,
                                          params.output_size, params.output_activation, device)
                     for _ in range(params.flock_size)]
        self.optimizers = [Adam(self.nets[i].parameters(), lr=params.learning_rate) for i in range(params.flock_size)]
        self._loss_func = params.output_activation.get_loss()

    def train(self, loaders: List[DataLoader], should_learn: List[bool]):
        """Trains the networks with the provided mini batches.

        Args:
            loaders: The training data loaders

        Returns:
            Nothing.
        """
        if self._params.output_activation == OutputActivation.SOFTMAX:
            preprocess_target = lambda t: t.argmax(dim=1)
        else:
            preprocess_target = lambda t: t

        # for loader, net, optimizer, _ in filter(itemgetter(4), zip(loaders, self.nets, self.optimizers, should_learn)):
        for loader, net, optimizer, s_l in zip(loaders, self.nets, self.optimizers, should_learn):
            net.train()
            #TODO: the loader returns just one mini-batch (sampled with replacement), so the for-loop here is a little confusing
            if s_l:
                for input_batch, target_batch in loader:
                    output = net.forward(input_batch)
                    target_batch = preprocess_target(target_batch)
                    loss = self._loss_func(output, target_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()  # apply gradients

    def forward_pass(self, inputs: torch.Tensor, outputs: torch.Tensor):
        """
        Runs the networks on a batch of points for each net.

        Args:
            inputs: [flock_size, batch_size, input_size]
            outputs: [flock_size, batch_size, output_size]

        Returns:
            Writes the result into the outputs tensor.
        """

        batch_size = inputs.shape[1]
        inputs_flattened = inputs.view(self._params.flock_size, batch_size, -1)

        for i, (net, input_data) in enumerate(zip(self.nets, inputs_flattened)):
            net.eval()
            outputs[i] = net.forward(input_data).view(outputs.shape[1:]).detach()

    def set_learning_rate(self, new_value: float):
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_value

    def test(self, inputs: torch.Tensor, targets: torch.Tensor, device='cuda') -> torch.Tensor:
        """Tests the networks on a set of inputs and targets.

        Args:
            inputs: The input data (flock_size, batch_size, input_size)
            targets: The targets (flock_size, batch_size, target_size)
            device: Device (cpu or cuda) for the output tensor

        Returns:
            Errors for each neural network (flock_size)
        """
        batch_size = inputs.shape[1]
        outputs = torch.zeros(self._params.flock_size, batch_size, self.output_size, device=device)
        self.forward_pass(inputs, outputs)
        return self.compute_errors_batch(outputs, targets)

    def _validate_inputs(self, input_batch: torch.Tensor, targets: torch.Tensor, learning_coefficients: torch.Tensor):

        batch_size = input_batch.shape[1]
        validate_predicate(lambda: input_batch.shape == (self.flock_size, batch_size, self.input_size))
        validate_predicate(lambda: targets.shape == (self.flock_size, batch_size, self.output_size))
        validate_predicate(lambda: learning_coefficients.shape == (self.flock_size, batch_size, 1))

