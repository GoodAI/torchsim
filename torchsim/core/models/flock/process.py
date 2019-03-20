from abc import ABC, abstractmethod
from typing import List, TypeVar

import torch

from torchsim.core.models.flock.buffer import Buffer
from torchsim.core.utils.tensor_utils import scatter_, gather_from_dim


TBuffer = TypeVar('TBuffer', bound=Buffer)


class Process(ABC):
    def __init__(self, indices: torch.Tensor, do_subflocking: bool):
        self.indices = indices
        self._do_subflocking = do_subflocking
        self._tensor_indices = self.indices.view(-1)
        self._flock_size = indices.size()[0]
        self._read_write_tensors = []
        self._all_read_tensors = []
        self._all_read_tensors = []
        self._buffer = None

    def _read_expanded(self, tensor: torch.Tensor) -> torch.Tensor:
        assert not self._is_in(tensor, self._all_read_tensors), "[_read] tensor has already been subflocked."
        subtensor = tensor.expand(self._flock_size, *list(tensor.size()[1:]))
        self._all_read_tensors.append(subtensor)
        return subtensor

    def _read(self, tensor: torch.Tensor) -> torch.Tensor:
        assert not self._is_in(tensor, self._all_read_tensors), "[_read] tensor has already been subflocked."
        if self._do_subflocking:
            subtensor = gather_from_dim(tensor, self._tensor_indices, 0)
        else:
            subtensor = tensor
        self._all_read_tensors.append(subtensor)
        return subtensor

    def _read_write(self, tensor: torch.Tensor) -> torch.Tensor:
        assert not self._is_in(tensor, self._all_read_tensors), "[_read_write] tensor has already been subflocked."
        if self._do_subflocking:
            subtensor = gather_from_dim(tensor, self._tensor_indices, 0)
        else:
            subtensor = tensor
        self._read_write_tensors.append((subtensor, tensor))
        self._all_read_tensors.append(tensor)
        return subtensor

    def _get_buffer(self, buffer: TBuffer) -> TBuffer:
        assert self._buffer is None, "buffer has already been subflocked."
        if self._do_subflocking:
            buffer.set_flock_indices(self.indices)

        self._buffer = buffer
        return buffer

    def run_and_integrate(self):
        self.run()
        if self._do_subflocking:
            self.integrate()

    @staticmethod
    def _is_in(tensor: torch.Tensor, items: List[torch.Tensor]) -> bool:
        for t in items:
            if tensor is t:
                return True
        return False

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _check_dims(self, *args):
        pass

    def integrate(self):
        for subtensor, tensor in self._read_write_tensors:
            scatter_(subtensor, tensor, self.indices)

        if self._buffer is not None:
            self._buffer.unset_flock_indices()
