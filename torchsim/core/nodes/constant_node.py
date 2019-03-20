from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator


class ConstantNodeUnit(Unit):
    def __init__(self, creator: TensorCreator, shape, constant):
        super().__init__(creator.device)
        self.output = creator.full(shape, fill_value=constant, dtype=self._float_dtype, device=creator.device)

    def step(self):
        pass


class ConstantNodeOutputs(MemoryBlocks):

    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")

    def prepare_slots(self, unit: ConstantNodeUnit):
        self.output.tensor = unit.output


class ConstantNode(WorkerNodeBase[EmptyInputs, ConstantNodeOutputs]):
    """ ConstantNode outputs tensor filled with one value.

    Dimension of output tensor must be specified on creation.
    Constant can be a non-number like ``float('nan')``.
    """

    outputs: ConstantNodeOutputs

    def __init__(self, shape, constant=0, name="Const"):
        if type(shape) is int:
            shape = (shape,)
        super().__init__(name=f"{name} {constant} [{tuple(shape)}]", outputs=ConstantNodeOutputs(self))
        self._constant = constant

        self._shape = shape

    def _create_unit(self, creator: TensorCreator):
        return ConstantNodeUnit(creator, self._shape, self._constant)

    def _step(self):
        self._unit.step()
