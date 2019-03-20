from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, TYPE_CHECKING
import torch

from torchsim.core.exceptions import ShouldNotBeCalledException, TensorNotSetException
from torchsim.core.models.flock.buffer import BufferStorage
from torchsim.core.utils.tensor_utils import view_dim_as_dims
from torchsim.gui.observers.buffer_observer import BufferObserver
from torchsim.gui.observers.memory_block_observer import MemoryBlockObserver

if TYPE_CHECKING:
    from torchsim.core.graph.node_base import NodeBase
    from torchsim.core.graph.connection import Connection


class SlotBase(ABC):
    name: str
    owner: 'NodeBase'

    def __init__(self, owner: 'NodeBase', name: str):
        self.name = name
        self.owner = owner

    @property
    @abstractmethod
    def tensor(self):
        pass

    def get_observable(self):
        return MemoryBlockObserver(self)

    @abstractmethod
    def clear(self):
        """Release allocated memory"""
        pass


class OutputSlotBase(SlotBase, ABC):
    """A slot which is connected to 0-N InputSlots via connections."""
    connections: List['Connection']

    def __init__(self, owner, name):
        super().__init__(owner, name)

        self.connections = []

    def add_connection(self, connection: 'Connection'):
        self.connections.append(connection)

    def remove_connection(self, connection: 'Connection'):
        self.connections.remove(connection)


class NoneSlot(SlotBase):
    @property
    def tensor(self):
        return None

    def clear(self):
        pass


class InputSlot(SlotBase):
    """A slot which is connected to an OutputSlot via a connection."""
    connection: 'Connection' = None

    @property
    def tensor(self):
        if self.connection is not None:
            return self.source.tensor

        return None

    @property
    def source(self) -> OutputSlotBase:
        return self.connection.source

    def clear(self):
        # Input slot is just a reference, no need to free it
        pass


class MemoryBlock(OutputSlotBase):
    """Acts as an output slot with an assigned tensor."""
    _tensor: torch.Tensor

    def __init__(self, owner: Optional['NodeBase'] = None, name: str = ''):
        super().__init__(owner, name)
        self._tensor = None

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, value):
        self._tensor = value

    @property
    def shape(self) -> List[int]:
        return list(self.tensor.shape)

    def reshape_tensor(self, shape: Tuple[int, ...], dim: int = -1):
        """Replaces self.tensor with a view, deriving the shape from the tensor by replacing the `dim` by `shape`.

        If dim == -1, this reshapes the whole tensor.
        It expects the tensor to be already set.
        """
        if self._tensor is None:
            raise TensorNotSetException("Tensor has to be set in order to call `reshape_tensor`.")

        self._tensor = view_dim_as_dims(self._tensor, shape, dim)

    def clear(self):
        self._tensor = None


class BufferMemoryBlock(MemoryBlock):
    _buffer_storage: BufferStorage = None

    def __init__(self, owner: 'NodeBase' = None, name: str = None):
        super().__init__(owner, name)

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, value):
        raise ShouldNotBeCalledException('Never set tensor of BufferMemoryBlock directly, use buffer property instead')

    @property
    def buffer(self):
        return self._buffer_storage

    @buffer.setter
    def buffer(self, value):
        self._buffer_storage = value
        self._tensor = self._buffer_storage.stored_data

    def get_observable(self):
        return BufferObserver(self)


class GroupVirtualOutputSlot(OutputSlotBase, ABC):
    input: InputSlot

    @property
    def tensor(self):
        return self.input.tensor

    def clear(self):
        # Virtual slot is just a reference, no need to free it
        pass


class GroupInputSlot(InputSlot):
    """An input of the NodeGroup.

    A slot which serves as an input to the owning NodeGroup and as a source for input connections of the inner nodes.
    """

    class GroupInputOutputSlot(GroupVirtualOutputSlot):
        """A virtual output slot which redirects tensor queries to the associated input slot."""
        def __init__(self, owner, name, input_slot):
            super().__init__(owner, name)
            self.input = input_slot

    output: GroupInputOutputSlot = None

    def __init__(self, owner, name):
        super().__init__(owner, name)

        self.output = self.GroupInputOutputSlot(owner, f"{name}.Output", self)


class GroupOutputSlot(GroupVirtualOutputSlot, ABC):
    """An output of the NodeGroup.

    A slot which serves as an output of the owning NodeGroup and as a destination for the output connections of
    the inner nodes.
    """
    def __init__(self, owner, name):
        self.input = InputSlot(owner, f"{name}.Input")
        super().__init__(owner, name)
