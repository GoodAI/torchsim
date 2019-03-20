from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator


class RandomUnitStub(Unit):
    def __init__(self, creator: TensorCreator):
        super().__init__('cpu')
        self.output = creator.randn((1,))

    def step(self):
        self.output.random_()


class RandomNodeOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create('output')

    def prepare_slots(self, unit: RandomUnitStub):
        self.output.tensor = unit.output


class RandomNodeStub(WorkerNodeBase):
    outputs: RandomNodeOutputs
    _unit: RandomUnitStub

    def __init__(self):
        super().__init__(outputs=RandomNodeOutputs(self))

    def _create_unit(self, creator: TensorCreator) -> Unit:
        return RandomUnitStub(creator)

    def _step(self):
        self._unit.step()
