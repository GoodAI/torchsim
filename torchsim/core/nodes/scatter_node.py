import torch

from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator


class ScatterNodeUnit(Unit):
    def __init__(self, creator: TensorCreator, device, mapping: torch.Tensor, output_shape, dimension: int):
        super().__init__(device)
        self.dimension = dimension
        self._mapping = mapping
        self.output = creator.zeros(output_shape, dtype=self._float_dtype, device=device)

    def step(self, input_tensor: torch.Tensor):
        self.output.fill_(0)
        self.output.scatter_(self.dimension, self._mapping, input_tensor)


class ScatterNodeInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create("Input")


class ScatterNodeOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")

    def prepare_slots(self, unit: ScatterNodeUnit):
        self.output.tensor = unit.output


class ScatterNode(WorkerNodeBase[ScatterNodeInputs, ScatterNodeOutputs]):
    """Fixed scatter to zero vector (mapping is set on initialization).

    If output shape is larger than mapping, missing values remain 0.
    """
    inputs: ScatterNodeInputs
    outputs: ScatterNodeOutputs

    def __init__(self, mapping, output_shape, dimension=0, device="cuda", name=""):
        super().__init__(name=name, inputs=ScatterNodeInputs(self), outputs=ScatterNodeOutputs(self))
        self._dimension = dimension
        self._output_shape = output_shape
        if type(mapping) is not torch.Tensor:
            mapping = torch.tensor(mapping, device=device, dtype=torch.long)
        if mapping.type() != 'torch.LongTensor' and mapping.type() != 'torch.cuda.LongTensor':
            raise ValueError("mapping in ScatterNode must be list of ints or torch.LongTensor.")
        self._mapping = mapping
        self._device = device

    def _check_slots(self):
        super()._check_slots()
        if torch.max(self._mapping) >= self._output_shape[self._dimension] or torch.min(self._mapping) < 0:
            raise ValueError("mapping in ScatterNode targets outside of output tensor.")

    def _create_unit(self, creator: TensorCreator):
        return ScatterNodeUnit(creator, self._device, self._mapping, self._output_shape, self._dimension)

    def _step(self):
        self._unit.step(self.inputs.input.tensor)
