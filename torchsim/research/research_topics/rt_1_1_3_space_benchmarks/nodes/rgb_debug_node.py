import logging

import torch

from torchsim.core.graph.slot_container import Inputs, MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator

logger = logging.getLogger(__name__)


class RgbDebugUnit(Unit):

    def __init__(self, creator: TensorCreator, input_dims, device, channel_first=False):
        super().__init__(device)
        self.creator = creator
        self._input_dims = input_dims
        self._channel_first = channel_first

        self.r_output_tensor = creator.zeros(input_dims, device=device, dtype=self._float_dtype)
        self.g_output_tensor = creator.zeros(input_dims, device=device, dtype=self._float_dtype)
        self.b_output_tensor = creator.zeros(input_dims, device=device, dtype=self._float_dtype)
        self.concat_output_tensor = creator.zeros([input_dims[0]*3] + list(input_dims[1:]), device=device, dtype=self._float_dtype)

    def step(self, data: torch.Tensor):

        if self._channel_first:
            if data.shape[0] != 3:
                logger.error("Incorrect dimension of the input, expected format of data is [channels=3,Y,X]")
                return
        else:
            if data.shape[2] != 3:
                logger.error("Incorrect dimension of the input, expected format of data is [Y,X,channels=3]")
                return

        if self._channel_first:
            input_data = data.permute(1, 2, 0)  # not a deep copy
        else:
            input_data = data

        tmp = input_data.clone()
        tmp[:, :, 1] = -1
        tmp[:, :, 2] = -1
        self.r_output_tensor.copy_(tmp)

        tmp = input_data.clone()
        tmp[:, :, 0] = -1
        tmp[:, :, 2] = -1
        self.g_output_tensor.copy_(tmp)

        tmp = input_data.clone()
        tmp[:, :, 0] = -1
        tmp[:, :, 1] = -1
        self.b_output_tensor.copy_(tmp)

        tmp = torch.cat([
            self.r_output_tensor,
            self.g_output_tensor,
            self.b_output_tensor,
        ])
        self.concat_output_tensor.copy_(tmp)

class RgbDebugInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create("Input")


class RgbDebugOutputs(MemoryBlocks):

    def __init__(self, owner):
        super().__init__(owner)
        self.r_output = self.create("r_output")
        self.g_output = self.create("g_output")
        self.b_output = self.create("b_output")
        self.concat_output = self.create("concat_output")

    def prepare_slots(self, unit: RgbDebugUnit):
        self.r_output.tensor = unit.r_output_tensor
        self.g_output.tensor = unit.g_output_tensor
        self.b_output.tensor = unit.b_output_tensor
        self.concat_output.tensor = unit.concat_output_tensor


class RgbDebugNode(WorkerNodeBase[RgbDebugInputs, RgbDebugOutputs]):
    """
    Just for debugging purposes: receives RGB image [sy, sx, 3]
    and splits into 3 outputs [r_output, g_output, b_output],
    where each of them has the same dimension as input, but only one channel contains values
    """

    _unit: RgbDebugUnit
    inputs: RgbDebugInputs
    outputs: RgbDebugOutputs

    def __init__(self, input_dims, device='cuda', channel_first=False):
        """
        Instance of the Node
        Args:
            input_dims: expected dimensions of the input
            device:
            channel_first: if True, then expected input is [channels=3, Y,X], otherwise [Y,X,channels=3]
        """
        super().__init__(name="RgbDebugUnit",
                         inputs=RgbDebugInputs(self),
                         outputs=RgbDebugOutputs(self))

        self.input_dims = input_dims
        self.device = device
        self.channel_first = channel_first

    def _create_unit(self, creator: TensorCreator) -> RgbDebugUnit:
        return RgbDebugUnit(creator,
                            input_dims=self.input_dims,
                            device=self.device,
                            channel_first=self.channel_first)

    def _step(self):
        self._unit.step(self.inputs.input.tensor)