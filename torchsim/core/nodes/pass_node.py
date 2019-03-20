import torch

from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.node_base import NodeValidationException
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator


class PassNodeUnit(Unit):
    def __init__(self, shape, creator: TensorCreator):
        super().__init__(creator.device)
        self.output = creator.zeros(shape, dtype=self._float_dtype, device=self._device)

    def step(self, tensor: torch.Tensor):
        self.output.copy_(tensor)


class PassNodeInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = (self.create("Input"))


class PassNodeOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = (self.create("Output"))

    def prepare_slots(self, unit: PassNodeUnit):
        self.output.tensor = unit.output


class PassNode(WorkerNodeBase[PassNodeInputs, PassNodeOutputs]):
    """This node fixes its output dimensions and provides zeros before it is called first time.

    Can be used in such a circular topology where the dims cannot be derived automatically.
    Put it after backward (low_priority) edge.
    """
    inputs: PassNodeInputs
    outputs: PassNodeOutputs

    def __init__(self, output_shape, name="Pass"):
        if type(output_shape) is int:
            output_shape = (output_shape,)
        super().__init__(inputs=PassNodeInputs(self), outputs=PassNodeOutputs(self),
                         name=f"{name} [{tuple(output_shape)}]")
        self._output_shape = output_shape

    def validate(self):
        if tuple(self.inputs.input.tensor.shape) != tuple(self._output_shape):
            raise NodeValidationException(f"In PassNode, input {self.inputs.input.tensor.shape} must equal declared "
                                          f"output {self._output_shape}.")

    def _create_unit(self, creator: TensorCreator):
        return PassNodeUnit(self._output_shape, creator)

    def _step(self):
        self._unit.step(self.inputs.input.tensor)
