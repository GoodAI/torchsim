from enum import Enum
from typing import List

import torch
from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.utils.tensor_utils import normalize_probs_, clamp_tensor, id_to_one_hot
from torchsim.gui.observables import ObserverPropertiesItem
from torchsim.utils.list_utils import dim_prod


class ToOneHotMode(Enum):
    RANDOM = 0
    MAX = 1


class ToOneHotUnit(Unit):
    def __init__(self, creator: TensorCreator, input_tensor_shape: torch.Size, mode: ToOneHotMode):
        super().__init__(creator.device)
        self.mode = mode
        self._last_dim = input_tensor_shape[-1]

        self.output = creator.zeros(input_tensor_shape, dtype=self._float_dtype, device=creator.device)
        self.normalized_dist = creator.zeros(input_tensor_shape, dtype=self._float_dtype, device=creator.device)
        self._zeros = creator.zeros((dim_prod(input_tensor_shape, end_dim=-2), 1), dtype=self._float_dtype, device=creator.device)

    def step(self, input_tensor: torch.Tensor):

        input_flattened = input_tensor.view(-1, self._last_dim)

        if self.mode == ToOneHotMode.RANDOM:
            # noinspection PyUnresolvedReferences
            minimum, _ = input_flattened.min(dim=1, keepdim=True)
            clamped = clamp_tensor(minimum, max=self._zeros)
            self.normalized_dist.copy_((input_flattened - clamped).view(input_tensor.shape))
            normalize_probs_(self.normalized_dist, dim=-1, add_constant=self.normalized_dist.sum() == 0)

            x = torch.multinomial(self.normalized_dist.view(-1, self._last_dim), 1)
        elif self.mode == ToOneHotMode.MAX:
            _, x = torch.max(input_flattened, dim=1)
        else:
            raise NotImplemented()

        one_hot = id_to_one_hot(x, self._last_dim)
        self.output.copy_(one_hot.view(self.output.shape))


class ToOneHotInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create("Input")


class ToOneHotOutputs(MemoryBlocks):

    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")

    def prepare_slots(self, unit: ToOneHotUnit):
        self.output.tensor = unit.output


class ToOneHotMemoryBlocks(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.normalized_dist = self.create("normalized")

    def prepare_slots(self, unit: ToOneHotUnit):
        self.normalized_dist.tensor = unit.normalized_dist


class ToOneHotNode(WorkerNodeBase[ToOneHotInputs, ToOneHotOutputs]):
    """Converts a vector of float values into a one-hot representation.

    It is done by shifting the vector to non-negative values and then sampling with probability
    proportional to the values of each element.
    """
    _mode: ToOneHotMode
    inputs: ToOneHotInputs
    outputs: ToOneHotOutputs

    @property
    def mode(self) -> ToOneHotMode:
        return self._mode

    @mode.setter
    def mode(self, value):
        if self._unit is not None:
            self._unit.mode = value

        self._mode = value

    def __init__(self, mode: ToOneHotMode = ToOneHotMode.RANDOM, name="ToOneHot"):
        super().__init__(name=name, inputs=ToOneHotInputs(self), outputs=ToOneHotOutputs(self))
        self._mode = mode

    def _create_unit(self, creator: TensorCreator):
        return ToOneHotUnit(creator, self.inputs.input.tensor.shape, self._mode)

    def validate(self):
        pass

    def _step(self):
        self._unit.step(self.inputs.input.tensor)

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return super().get_properties() + [self._prop_builder.auto("Mode", type(self).mode)]


