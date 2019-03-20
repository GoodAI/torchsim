from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import torch
import torch.nn as nn
from torchsim.core.models.expert_params import ParamsBase


class OutputActivation(Enum):
    SOFTMAX = auto()   # good for multi-class classification
    TANH = auto()      # regression for bounded within [-1, 1], but gradients vanish at the extremes :(
    IDENTITY = auto()  # regression for unbounded values
    SIGMOID = auto()

    def get_activation(self) -> nn.Module:
        if self == OutputActivation.SOFTMAX:
            return nn.Softmax()
        elif self == OutputActivation.TANH:
            return nn.Tanh()
        elif self == OutputActivation.IDENTITY:
            return nn.Sequential()  # i.e. identity
        elif self == OutputActivation.SIGMOID:
            return nn.Sigmoid()
        else:
            raise ValueError(f'Unknown output activation value: {self}')

    def get_loss(self) -> nn.Module:
        if self == OutputActivation.SOFTMAX:
            return nn.CrossEntropyLoss()
        elif self == OutputActivation.TANH or self == OutputActivation.IDENTITY or self == OutputActivation.SIGMOID:
            return nn.MSELoss()
        else:
            raise ValueError(f'Unknown output activation value: {self}')


@dataclass
class NeuralNetworkFlockParams(ParamsBase):
    flock_size: int = 1  # number of NNs to create
    input_size: int = 1
    hidden_size: int = 20
    n_hidden_layers: int = 1
    output_size: int = 1
    learning_rate: float = 0.001
    mini_batch_size: int = 30
    output_activation: OutputActivation = OutputActivation.SOFTMAX

    # If the maximum of all the data learning coefficients for given network is lower than this,
    # the network doesn't learn at all.
    coefficients_minimum_max: float = 0.0


class NeuralNetworkFlockTypes(Enum):
    """Possible types of the networks."""
    MLP = 1


class NeuralNetworkFlock:
    """An interface for the neural network flock for the network_flock_node"""

    @abstractmethod
    def train(self, loaders: List[DataLoader], should_learn: List[bool]):
        """Train each network using the provided data loaders."""
        pass

    @abstractmethod
    def forward_pass(self, inputs: torch.Tensor, outputs: torch.Tensor):
        """Compute forward pass of each network using one piece of data from `inputs`."""
        pass

    def set_learning_rate(self, new_value: float):
        """Updates the learning rate of each neural network."""
        pass

    @staticmethod
    def compute_errors(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the errors of each network for one output - correct_target pair.

        We currently use MSE for this. It is computed for each network separatelly

        Args:
            outputs: network outputs (flock_size, *output_shape)
            targets: targets (flock_size, *output_shape)

        Returns:
            Errors of the networks (flock_size)
        """
        return NeuralNetworkFlock.compute_errors_batch(outputs.unsqueeze(dim=1), targets.unsqueeze(dim=1))

    @staticmethod
    def compute_errors_batch(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the errors of the networks.

        We currently use MSE for this. It is computed for each network.

        Args:
            outputs: network outputs (flock_size, batch_size, *output_shape)
            targets: targets (flock_size, batch_size, *output_shape)

        Returns:
            Errors of the networks (flock_size)
        """

        diff = outputs - targets
        diff = diff.view(diff.shape[0], diff.shape[1], -1)

        squared_errors = diff.pow(2).sum(dim=2)  # Take the squared difference for each element
        mean_squared_errors = squared_errors.mean(dim=1)  # Take the average over each batch

        return mean_squared_errors

    def _validate_inputs(self, input_batch: torch.Tensor, targets: torch.Tensor, learning_coefficients: torch.Tensor):
        """Validate the shape of the input tensors."""

        pass

    def make_data_loaders(self,
                          input_batch: torch.Tensor,
                          targets: torch.Tensor,
                          learning_coefficients: torch.Tensor,
                          mini_batch_size: int,
                          coefficients_minimum_max) -> Tuple[List[DataLoader], List[bool]]:
        """Produces tensors with mini batches for training for each network.

        Args:
            coefficients_minimum_max: threshold for learning (a minimal value of the max(coefficients))
            input_batch: The input data (flock_size, batch_size, input_size)
            targets: The targets (flock_size, batch_size, target_size)
            learning_coefficients: The learning coefficients (how much should each datapoint be learned by each network)
            (flock_size, batch_size)
            mini_batch_size: The number of examples in each mini batch

        Returns:
            Data loaders for each network.
        """

        self._validate_inputs(input_batch, targets, learning_coefficients)

        train_loaders = []
        should_learn = []
        for nn_input, nn_targets, nn_weights in zip(input_batch, targets, learning_coefficients):
            data_set = TensorDataset(nn_input, nn_targets)
            # We remove the last singleton dimension and unsqueeze the weights here to sample the whole batch at once
            sampler = WeightedRandomSampler(nn_weights.squeeze(-1).unsqueeze(0), mini_batch_size)

            network_should_learn = nn_weights.max().item() > coefficients_minimum_max
            should_learn.append(network_should_learn)

            # train_loader = DataLoader(data_set, batch_size=mini_batch_size, shuffle=False, sampler=sampler)
            train_loader = DataLoader(data_set, batch_sampler=sampler)
            train_loaders.append(train_loader)
        return train_loaders, should_learn


