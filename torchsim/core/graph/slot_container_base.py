from abc import ABC, abstractmethod
from typing import List, Sized, TypeVar, Generic, TYPE_CHECKING

from torchsim.core.graph.slots import InputSlot, MemoryBlock, BufferMemoryBlock, SlotBase, OutputSlotBase
from torchsim.core.graph.unit import Unit

TSlot = TypeVar('TSlot', bound=SlotBase)

if TYPE_CHECKING:
    pass


class MemoryBlockInitializationException(Exception):
    pass


class NonExistentMemoryBlockException(Exception):
    pass


TSlotContainerBase = TypeVar('TSlotContainerBase', bound='SlotContainerBase')
TNodeBase = TypeVar('TNodeBase', bound='NodeBse')


class SlotContainerBase(Sized, Generic[TSlot, TNodeBase], ABC):
    """Holds memory blocks, usually those of a NodeBase.

    The blocks are stored in _blocks and can be accessed via integer indexing (the order is the same as order of
    creation).
    """
    _owner: TNodeBase
    _slots: List[TSlot]

    def __init__(self, owner: TNodeBase):
        self._owner = owner
        self._slots = []

    def __getitem__(self, index: int) -> TSlot:
        return self._slots[index]

    def __len__(self) -> int:
        return len(self._slots)

    def set_interpret_dims(self):
        """Setup the interpret dimensions of the memory blocks here if you want. They are then used for observation."""
        pass

    def check_preparation(self):
        pass

    def create(self, name: str) -> TSlot:
        slot = self._create_slot_instance(name)
        self._slots.append(slot)
        return slot

    def clear(self):
        """Release all allocated memory"""
        for slot in self._slots:
            slot.clear()

    @abstractmethod
    def _create_slot_instance(self, name: str) -> TSlot:
        pass


TInputSlot = TypeVar('TInputSlot', bound=InputSlot)


class InputsBase(SlotContainerBase[TInputSlot, TNodeBase], ABC):
    """Used as a base class for inputs to any node."""


class GenericInputsBase(InputsBase[InputSlot, TNodeBase]):
    """Standard inputs used in all WorkerNodeBase implementations."""

    def _create_slot_instance(self, name) -> InputSlot:
        return InputSlot(self._owner, name)


TOutputSlot = TypeVar('TOutputSlot', bound=OutputSlotBase)


class OutputsBase(SlotContainerBase[TOutputSlot, TNodeBase], ABC):
    def prepare_slots(self, unit: Unit):
        """Prepare slots after the unit is created."""
        pass


class GenericMemoryBlocks(OutputsBase[MemoryBlock, TNodeBase]):
    """A container for allocated memory blocks which are not normally used as outputs of the node."""

    def _create_slot_instance(self, name) -> MemoryBlock:
        return MemoryBlock(self._owner, name)

    def create_buffer(self, name: str):
        block = BufferMemoryBlock(self._owner, name)
        self._slots.append(block)
        return block

    def index(self, memory_block):
        return self._slots.index(memory_block)

    @abstractmethod
    def prepare_slots(self, unit: Unit):
        """Connect tensors from the unit with the memory blocks here."""
        pass

    def check_preparation(self):
        for memory_block in self._slots:
            if memory_block.tensor is None:
                node_info = ""
                if self._owner is not None:
                    node_info = f" of node '{self._owner.name}'"

                message = f"Memory block '{memory_block.name}'{node_info} does not have a tensor"
                raise MemoryBlockInitializationException(message)
