from typing import Optional, TYPE_CHECKING, Callable

import torch
from torchsim.core.global_settings import SimulationThreadRunner
from torchsim.core.memory.tensor_creator import TensorSurrogate
from torchsim.gui.observers.tensor_observable import TensorObservable
from torch import Tensor

if TYPE_CHECKING:
    from torchsim.core.graph.slots import SlotBase


def is_valid_tensor(tensor: torch.Tensor):
    if tensor is None:
        return False
    elif type(tensor) is TensorSurrogate:
        return False
    else:
        return True


class MemoryBlockObserver(TensorObservable):
    """Basic observer for a MemoryBlock, converts everything into 2D tensor (using the interpret.shape)."""
    _block: 'SlotBase'

    def __init__(self, block: 'SlotBase'):
        super().__init__()
        self._block = block

    def get_tensor(self) -> Optional[torch.Tensor]:
        # torch.cuda.synchronize()
        # clone tensor so it is consistent during coloring/tiling processing (otherwise strange artifacts appears)
        if self._block.owner.is_initialized():
            return self._block.tensor
        else:
            return None


class CustomTensorObserver(TensorObservable):
    _tensor_provider: Callable[[], Tensor]
    _last_value: Optional[torch.Tensor] = None

    def __init__(self, tensor_provider: Callable[[], torch.Tensor]):
        super().__init__()
        self._tensor_provider = tensor_provider

    def get_tensor(self) -> Optional[torch.Tensor]:
        SimulationThreadRunner.instance().run_in_simulation_thread(self._compute_value)
        # return self._tensor_provider()
        return self._last_value

    def _compute_value(self):
        self._last_value = self._tensor_provider()

# class OneExpertMemoryBlockObserver(BaseMemoryBlockObserver):
#     """Basic observer for one expert in a MemoryBlock, converts all into 2D tensor (using the interpret.shape)."""
#
#     def __init__(self, node: 'NodeBase', expert_no: int, block: MemoryBlock):
#         super().__init__(node, block)
#         self._expert_no = expert_no
#
#         # convert the ND tensor into 2D tensor by merging all the dimensions except the last one (columns)
#         # TODO (Feat) should be improved with custom shapes in the future
#
#         dims = self._interpret_shape.copy()
#         dims.pop(0)  # clone and remove the expert_id dimension
#
#         # self._observable_dims = self._squash_all_dims_but_last(dims)
#
#         # self._auto_upscale(self._observable_dims, self.MIN_OBSERVER_SIZE)
#
#     def get_tensor(self) -> torch.Tensor:
#         torch.cuda.synchronize()
#         tensor = self._block.tensor
#         if tensor is None:
#             return torch.full((1, 1), float('nan'))
#         elif type(tensor) is TensorSurrogate:
#             return torch.full(tensor.size()[1:], float('nan'))  # omit the first dimension
#         else:
#             return tensor.to('cpu').data[self._expert_no]  # first dimension is always expert_id in the flock
