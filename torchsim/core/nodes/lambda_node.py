import torch
from typing import List, Tuple, Any, Callable, Union

from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator


class LambdaNodeUnit(Unit):
    def __init__(self, func, output_shapes, creator: TensorCreator):
        super().__init__(creator.device)
        self.func = func
        self.outputs = []
        for shape in output_shapes:
            self.outputs.append(creator.zeros(shape, dtype=self._float_dtype, device=self._device))

    def step(self, tensors: List[torch.Tensor], memory_tensors: List[torch.Tensor]):
        if len(memory_tensors) == 0:
            self.func(inputs=tensors, outputs=self.outputs)
        else:
            self.func(inputs=tensors, outputs=self.outputs, memory=memory_tensors)


class NInputs(Inputs):
    def __init__(self, owner, n):
        super().__init__(owner)
        for i in range(n):
            self.create(f"Input {i}")


class NOutputs(MemoryBlocks):
    def __init__(self, owner, n):
        super().__init__(owner)
        for i in range(n):
            self.create(f"Output {i}")

    def prepare_slots(self, unit: LambdaNodeUnit):
        for i, output in enumerate(unit.outputs):
            self[i].tensor = output


class LambdaNode(WorkerNodeBase[NInputs, NOutputs]):
    """ LambdaNode computes custom function on fixed number of inputs and outputs.

    The custom function is of following type:
        ``def custom_f(inputs: List[torch.Tensor], outputs: List[torch.Tensor], [memory: List[torch.Tensor]]): ...``

    The outputs must be filled using ``outputs[0].copy_(r)`` and the inputs shall not be changed.

    The output shapes must be specified on creation.
    """

    inputs: NInputs
    outputs: NOutputs
    _unit: LambdaNodeUnit

    def __init__(self, func: Union[Callable[[List[torch.Tensor], List[torch.Tensor]], Any],
                                   Callable[[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]], Any]],
                 n_inputs, output_shapes: List[Tuple], memory_shapes: List[Tuple] = (), name="Lambda"):
        super().__init__(inputs=NInputs(self, n_inputs), outputs=NOutputs(self, len(output_shapes)), name=name)
        self._output_shapes = output_shapes
        self._func = func
        if memory_shapes is None or len(memory_shapes) == 0:
            self.memory = []
        else:
            self.memory = [torch.zeros(memory_shape) for memory_shape in memory_shapes]

    def _create_unit(self, creator: TensorCreator):
        return LambdaNodeUnit(self._func, self._output_shapes, creator)

    def _step(self):
        self._unit.step([memory_block.tensor for memory_block in self.inputs], self.memory)

    def change_function(self, func: Callable[[List[torch.Tensor], List[torch.Tensor]], Any]):
        """This is safe to call in middle of the simulation as long as the function operate over the same dims."""
        self._unit.func = func
