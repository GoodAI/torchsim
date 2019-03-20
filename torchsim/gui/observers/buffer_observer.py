import logging
import math
from typing import List, NamedTuple, Optional, TYPE_CHECKING

import torch
from torchsim.gui.observers.memory_block_observer import is_valid_tensor
from torchsim.gui.observers.tensor_observable import TensorObservable, TensorObservableParams, TensorObservableData, \
    TensorViewProjection, dummy_tensor_observable_data, sanitize_tensor


if TYPE_CHECKING:
    from torchsim.core.graph.slots import BufferMemoryBlock


logger = logging.getLogger(__name__)


class BufferObserverParams(TensorObservableParams):
    highlight_type: int


class BufferObserverData(NamedTuple):
    tensor_data: TensorObservableData
    current_ptr: List[int]


class BufferObserver(TensorObservable):
    """Buffer observer - observe MemoryBlock and highlight current positions."""
    _buffer_memory_block: 'BufferMemoryBlock'
    _observable_dims: List[int]

    def __init__(self, buffer_memory_block: 'BufferMemoryBlock'):
        super().__init__()
        # TODO not nice to redefine _tensor_view_projection here
        self._tensor_view_projection = TensorViewProjection(is_buffer=True)
        self._buffer_memory_block = buffer_memory_block

    def get_tensor(self) -> Optional[torch.Tensor]:
        """Get tensor to be displayed."""
        if not self._buffer_memory_block.owner.is_initialized():
            return None

        return self._buffer_memory_block.buffer.stored_data if self._buffer_memory_block.buffer is not None else None

    def get_data(self) -> BufferObserverData:
        self._tensor = self.get_tensor()
        if self._tensor is not None:
            self._tensor = sanitize_tensor(self._tensor)
            tensor, projection_params = self._tensor_view_projection.transform_tensor(self._tensor, self._is_rgb)
            self._update_scale_to_respect_minimum_size(tensor)

            current_ptr = []
            if self._buffer_memory_block.buffer is not None:
                ptr_tensor = self._buffer_memory_block.buffer.current_ptr
                if is_valid_tensor(ptr_tensor):
                    flock_size = ptr_tensor.numel()
                    buffer_size = math.floor(projection_params.count / flock_size)
                    offsets = torch.arange(0, buffer_size * flock_size, buffer_size, device=ptr_tensor.device)
                    current_ptr = ptr_tensor.add(offsets).to("cpu").numpy().tolist()

            tensor_data = TensorObservableData(tensor, TensorObservableParams(
                scale=self._scale,
                projection=projection_params
            ))
            result = BufferObserverData(tensor_data, current_ptr=current_ptr)
        else:
            result = BufferObserverData(dummy_tensor_observable_data(), current_ptr=[0])
        return result
        # return BufferObserverData()
