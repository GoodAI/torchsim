
from abc import abstractmethod
from typing import Tuple

import torch

from torchsim.core import FLOAT_NAN
from torchsim.core.memory.tensor_creator import TensorCreator


class DelayBuffer:
    """The delay buffer delays the tensors for an easy access later, can have either delay 0 of 1 for now"""

    @abstractmethod
    def push(self, data: torch.Tensor):
        pass

    @abstractmethod
    def read(self) -> torch.Tensor:
        pass


class ZeroStepTensorDelayBuffer(DelayBuffer):
    """Buffer which does not delay the input"""

    _tensor: torch.Tensor

    def push(self, data: torch.Tensor):
        self._tensor = data

    def read(self) -> torch.Tensor:
        return self._tensor


class OneStepTensorDelayBuffer(DelayBuffer):
    """Circular buffer which is able to delay the tensor by one step"""

    _odd_tensor: torch.Tensor
    _even_tensor: torch.Tensor

    _is_last_written_even: bool
    _num_writes: int

    def __init__(self,
                 creator: TensorCreator,
                 tensor_shape: Tuple):

        self._odd_tensor = creator.full(tensor_shape, FLOAT_NAN, device=creator.device, dtype=creator.float)
        self._even_tensor = creator.full(tensor_shape, FLOAT_NAN, device=creator.device, dtype=creator.float)

        self._num_writes = 0

    def push(self, data: torch.Tensor):

        if self._num_writes % 2 == 0:
            written_to = self._odd_tensor
        else:
            written_to = self._even_tensor

        written_to.copy_(data)
        self._num_writes += 1

    def read(self) -> torch.Tensor:
        if self._num_writes % 2 == 0:
            return self._odd_tensor
        else:
            return self._even_tensor


def create_delay_buffer(creator: TensorCreator, is_delayed: bool, tensor_shape: Tuple) -> DelayBuffer:
    """A factory which creates the delay buffer (either delay 1 or 0)"""
    if is_delayed:
        return OneStepTensorDelayBuffer(creator, tensor_shape)
    return ZeroStepTensorDelayBuffer()
