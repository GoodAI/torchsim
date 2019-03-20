import torch

from torchsim.core.graph.slots import InputSlot, OutputSlotBase


class InversePassOutputPacket:
    """A class for holding data associated with a given output memory block to be used in the inverse projection.

    The data is a tensor of the same size as the tensor stored in memory_block, and it is the output that is going to be
    projected back into the input space of the node that memory_block belongs to.

    'data' is therefore something that could have come from the memory_block.
    """
    tensor: torch.Tensor
    slot: OutputSlotBase

    def __init__(self, data: torch.Tensor, output_slot: OutputSlotBase):
        self.tensor = data
        self.slot = output_slot


class InversePassInputPacket:
    tensor: torch.Tensor
    slot: InputSlot

    def __init__(self, data: torch.Tensor, input_slot: InputSlot):
        self.tensor = data
        self.slot = input_slot
