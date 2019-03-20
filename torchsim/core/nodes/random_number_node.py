import logging
from dataclasses import dataclass

import numpy as np

import torch
from torchsim.core import FLOAT_NAN
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver
from torchsim.core.utils.tensor_utils import id_to_one_hot
from torchsim.gui.observables import disable_on_runtime
from torchsim.gui.observer_system import ObserverPropertiesItem
from torchsim.gui.validators import *
from torchsim.utils.seed_utils import get_rand_generator_or_set_cuda_seed

logger = logging.getLogger(__name__)


class RandomNumberUnit(Unit):
    """Generate integers uniformly from <0,_value_range) and present in a scalar and one-hot representation."""

    _upper_bound: int
    _random: np.random

    _scalar_output: torch.Tensor
    _one_hot_output: torch.Tensor
    _current_value: torch.Tensor
    _step: int
    _next_generation: int

    def __init__(self,
                 creator: TensorCreator, lower_bound: int,
                 upper_bound: int, random: np.random, generate_new_every_n: int = 1,
                 generate_random_intervals=False):
        super().__init__(creator.device)

        self._generate_new_every_n = generate_new_every_n
        self._next_generation = generate_new_every_n
        self._step = -1
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._random = random
        self._random_generation_intervals = generate_random_intervals

        self._scalar_output = creator.full([1],
                                           fill_value=FLOAT_NAN, dtype=self._float_dtype, device=self._device)
        self._one_hot_output = creator.full([self._upper_bound],
                                            fill_value=FLOAT_NAN, dtype=self._float_dtype, device=self._device)

        self._current_value = creator.zeros((1,), dtype=creator.long)

    def step(self):
        self._step += 1
        if self._step % self._next_generation != 0:
            return
        self._step = 0
        if self._random_generation_intervals:
            self._next_generation = self._random.randint(low=1, high=self._generate_new_every_n + 1)
        self._current_value[0] = self._random.randint(low=self._lower_bound, high=self._upper_bound)
        self._scalar_output[0] = self._current_value[0]
        self._one_hot_output.copy_(id_to_one_hot(self._current_value, self._upper_bound).squeeze())

    def _save(self, saver: Saver):
        random_state = list(self._random.get_state())
        random_state[1] = [int(value) for value in random_state[1]]
        saver.description['_random_state'] = tuple(random_state)

        saver.description['_step'] = self._step
        saver.description['_next_generation'] = self._next_generation

    def _load(self, loader: Loader):
        self._random.set_state(loader.description['_random_state'])

        self._step = loader.description['_step']
        self._next_generation = loader.description['_next_generation']


class RandomNumberOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.scalar_output = self.create("Scalar")
        self.one_hot_output = self.create("One-hot")

    def prepare_slots(self, unit: RandomNumberUnit):
        self.scalar_output.tensor = unit._scalar_output
        self.one_hot_output.tensor = unit._one_hot_output


@dataclass
class RandomNumberNodeParams(ParamsBase):
    seed: int = 1
    lower_bound: int = 0
    upper_bound: int = 1
    generate_new_every_n: int = 1
    randomize_intervals: bool = False


class RandomNumberNode(WorkerNodeBase[EmptyInputs, RandomNumberOutputs]):
    """Generate integers uniformly from <lower_bound, upper_bound) and present in a scalar and one-hot representation.

    The numbers are generated on CPU by independent random generator
    (can be deterministic during the entire simulation).
    """
    _unit: RandomNumberUnit
    _params: RandomNumberNodeParams
    outputs: RandomNumberOutputs

    _seed: int
    _upper_bound: int

    def __init__(self, lower_bound: int = 0, upper_bound: int = 10, seed: int = None, name="RandomNumber",
                 generate_new_every_n: int = 1, randomize_intervals: bool = False):
        """Initialize.

        Args:
            generate_new_every_n: will skip n-1 steps
            randomize_intervals: will skip random number of steps between 0 and generate_new_every_n
        """
        super().__init__(name=name, outputs=RandomNumberOutputs(self))
        self._params = RandomNumberNodeParams(seed, lower_bound, upper_bound, generate_new_every_n, randomize_intervals)

    def _create_unit(self, creator: TensorCreator):
        random = get_rand_generator_or_set_cuda_seed(creator.device, self._params.seed)

        return RandomNumberUnit(creator, self._params.lower_bound, self._params.upper_bound, random,
                                self._params.generate_new_every_n, self._params.randomize_intervals)

    def _step(self):
        self._unit.step()

    @property
    def seed(self) -> int:
        return self._params.seed

    @seed.setter
    def seed(self, value: int):
        validate_positive_int(value)
        self._params.seed = value

    @property
    def lower_bound(self) -> int:
        return self._params.lower_bound

    @lower_bound.setter
    def lower_bound(self, value: int):
        validate_positive_with_zero_int(value)
        self._params.lower_bound = value

    @property
    def upper_bound(self) -> int:
        return self._params.upper_bound

    @upper_bound.setter
    def upper_bound(self, value: int):
        validate_positive_int(value)
        self._params.upper_bound = value

    @property
    def generate_new_every_n(self) -> int:
        return self._params.generate_new_every_n

    @generate_new_every_n.setter
    def generate_new_every_n(self, value: int):
        validate_positive_int(value)
        self._params.generate_new_every_n = value

    @property
    def randomize_intervals(self) -> bool:
        return self._params.randomize_intervals

    @randomize_intervals.setter
    def randomize_intervals(self, value):
        self._params.randomize_intervals = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
            self._prop_builder.auto('Seed', type(self).seed, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Lower bound', type(self).lower_bound, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Upper bound', type(self).upper_bound, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Generate new every N', type(self).generate_new_every_n,
                                    edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Randomize intervals', type(self).randomize_intervals,
                                    edit_strategy=disable_on_runtime)
        ]
