import logging
from typing import Callable

import torch

from torchsim.core import FLOAT_TYPE_CPU, FLOAT_NAN
from torchsim.core.models.flock import Process
from torchsim.core.utils.tensor_utils import change_dim, scatter_
from torchsim.gui.observers.tensor_observable import TensorObservable

logger = logging.getLogger(__name__)


class FlockProcessObservable(TensorObservable):
    _cached_tensor: torch.Tensor
    _process_provider: Callable[[], Process]
    _tensor_provider: Callable[[Process], torch.Tensor]

    def __init__(self,
                 flock_size: int,
                 process_provider: Callable[[], Process],
                 tensor_provider: Callable[[Process], torch.Tensor]):
        super().__init__()
        self._process_provider = process_provider
        self._tensor_provider = tensor_provider
        self._flock_size = flock_size
        self._cached_tensor = None

    def _init_tensor(self, current_tensor):
        if self._cached_tensor is None:
            dims = change_dim(current_tensor.shape, 0, self._flock_size)
            self._cached_tensor = torch.zeros(dims, dtype=current_tensor.dtype, device=current_tensor.device)

        self._clear_cached_tensor()

    def _clear_cached_tensor(self):
        if self._cached_tensor.dtype == FLOAT_TYPE_CPU:
            self._cached_tensor.fill_(FLOAT_NAN)
        else:
            self._cached_tensor.fill_(0)

    def get_tensor(self) -> torch.Tensor:
        # If a cached tensor exists, clear it.
        if self._cached_tensor is not None:
            self._clear_cached_tensor()

        process = self._process_provider()
        if process is None:
            # This will return None until a tensor is seen at least once.
            return self._cached_tensor

        current_tensor = self._tensor_provider(process)
        if current_tensor is None:
            return self._cached_tensor

        # If there is no cached tensor, create it.
        if self._cached_tensor is None:
            self._init_tensor(current_tensor)

        scatter_(current_tensor, self._cached_tensor, process.indices)

        return self._cached_tensor
