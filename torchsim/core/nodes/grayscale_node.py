import torch

from torchsim.core.graph.node_base import NodeValidationException
from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.utils.image_processing_utilities import ImageProcessingUtilities


class GrayscaleNodeUnit(Unit):
    def __init__(self, output_shape, creator: TensorCreator, squeeze_channel: bool):
        super().__init__(creator.device)
        self.output = creator.zeros(output_shape, dtype=self._float_dtype, device=self._device)
        self._squeeze_channel = squeeze_channel

    def step(self, tensor: torch.Tensor):
        luminance = ImageProcessingUtilities.rgb_to_grayscale(tensor, self._squeeze_channel)
        self.output.copy_(luminance)


class GrayscaleNodeInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = (self.create("Input"))


class GrayscaleNodeOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = (self.create("Output"))

    def prepare_slots(self, unit: GrayscaleNodeUnit):
        self.output.tensor = unit.output


class GrayscaleNode(WorkerNodeBase[GrayscaleNodeInputs, GrayscaleNodeOutputs]):
    """ GrayscaleNode computes luminance of the input X*Y*3 RGB image.

    The output shape is X*Y.
    """

    inputs: GrayscaleNodeInputs
    outputs: GrayscaleNodeOutputs
    _unit: GrayscaleNodeUnit

    def __init__(self, squeeze_channel: bool = False, name="Grayscale"):
        super().__init__(inputs=GrayscaleNodeInputs(self), outputs=GrayscaleNodeOutputs(self),
                         name=name)
        self._squeeze_channel = squeeze_channel

    def validate(self):
        if self.inputs.input is not None:
            if tuple(self.inputs.input.tensor.shape)[-1] != 3:
                raise NodeValidationException(f"In GrayscaleNode, the RGB input shape {self.inputs.input.tensor.shape} "
                                              f" must be end with 3 (the RGB channels).")

    def _create_unit(self, creator: TensorCreator):
        output_shape = self.inputs.input.tensor.shape[:-1]
        if not self._squeeze_channel:
            output_shape = output_shape + (1, )
        self._output_shape = tuple(output_shape)
        return GrayscaleNodeUnit(self._output_shape, creator, self._squeeze_channel)

    def _step(self):
        self._unit.step(self.inputs.input.tensor)
