from typing import Generic

from abc import abstractmethod

from torchsim.core.graph.node_base import NodeBase
from torchsim.core.graph.slot_container_base import GenericInputsBase, TSlotContainerBase, GenericMemoryBlocks, TSlot
from torchsim.core.graph.slots import BufferMemoryBlock
from torchsim.core.graph.unit import Unit


class MemoryBlocks(GenericMemoryBlocks[NodeBase]):
    @abstractmethod
    def prepare_slots(self, unit: Unit):
        """Connect tensors from the unit with the memory blocks here."""
        pass


class Inputs(GenericInputsBase[NodeBase]):
    pass


class SlotSection(Generic[TSlotContainerBase]):
    _container: TSlotContainerBase

    def __init__(self, container: TSlotContainerBase):
        # This init is important for multiple inheritance to work (when not present, not all init methods in hierarchy
        # are called
        self._container = container

    def create(self, name: str) -> TSlot:
        return self._container.create(name)


class MemoryBlocksSection(SlotSection[GenericMemoryBlocks[NodeBase]]):

    def create_buffer(self, name: str) -> BufferMemoryBlock:
        return self._container.create_buffer(name)


class InputsSection(SlotSection[Inputs]):
    pass
