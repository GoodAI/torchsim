from dataclasses import dataclass, astuple
from enum import Enum, IntEnum
from typing import Tuple

import numpy as np

import torch
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.gui.observables import ObserverPropertiesItem, disable_on_runtime
from torchsim.gui.validators import *
from torchsim.utils.param_utils import Size2D, Point2D
from torchsim.utils.seed_utils import get_rand_generator_or_set_cuda_seed


class MoveStrategy(Enum):
    DIRECT_TO_POINT = 0  # Point moves in the direction of the attractor
    # Point moves left,left/down,down,right for attractor in positions: left/up, left/down, right/down, right/up
    LIMITED_TO_LEFT_DOWN_QUADRANT = 1


@dataclass
class DatasetSimplePointGravityParams(ParamsBase):
    canvas_shape: Size2D
    point_pos: Point2D
    attractor_distance: int
    move_strategy: MoveStrategy


class States(IntEnum):
    BLANK = 0
    SHOW = 1
    MOVE = 2

    def next(self):
        last = States.MOVE
        v = self.value + 1
        return States(v) if v <= last.value else States.BLANK


class DatasetSimplePointGravityUnit(Unit):
    _frame_backbuffer: torch.Tensor
    rand_vector: np.array
    _random: np.random
    output_data: torch.Tensor
    _state: States
    move_strategy: MoveStrategy

    def __init__(self, creator: TensorCreator, params: DatasetSimplePointGravityParams, random: np.random):
        super().__init__(creator.device)
        self._random = random
        self.point_pos = params.point_pos
        self.attractor_distance = params.attractor_distance
        self.output_data = creator.zeros(*params.canvas_shape, device=self._device)
        self._state = States.BLANK
        self.move_strategy = params.move_strategy
        self._frame_backbuffer = creator.zeros_like(self.output_data)

    def step(self):
        self._frame_backbuffer.fill_(0)
        if self._state == States.BLANK:
            self.rand_vector = self._random.randint(0, 2, 2) * 2 - 1  # generate diagonal 2D vector
        elif self._state == States.SHOW:
            # Point
            self._frame_backbuffer[self.point_pos.y, self.point_pos.x] = 1
            # Attractor
            target_pos = np.add(self.rand_vector * self.attractor_distance, astuple(self.point_pos))
            self._frame_backbuffer[target_pos[0], target_pos[1]] = 0.5
        elif self._state == States.MOVE:
            # Point
            target_point = self._move_point(self.rand_vector, self.point_pos)
            self._frame_backbuffer[target_point.y, target_point.x] = 1
            # Attractor
            target_pos = np.add(self.rand_vector * self.attractor_distance, astuple(self.point_pos))
            self._frame_backbuffer[target_pos[0], target_pos[1]] = 0.5
        self._state = self._state.next()
        self.output_data.copy_(self._frame_backbuffer)

    def _move_point(self, rand_vector: np.array, point: Point2D) -> Point2D:
        if self.move_strategy == MoveStrategy.DIRECT_TO_POINT:
            return Point2D(*np.add(rand_vector, astuple(point)))
        elif self.move_strategy == MoveStrategy.LIMITED_TO_LEFT_DOWN_QUADRANT:
            rand_y, rand_x = rand_vector
            rand_y = max(rand_y, 0)
            # move directly down instead right/down
            if rand_y == 1 and rand_x == 1:
                rand_x = 0
            return Point2D(point.y + rand_y, point.x + rand_x)


class DatasetSimplePointGravityOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.data = self.create("Output_data")
        # self.label = self.create("Output_label")

    def prepare_slots(self, unit: DatasetSimplePointGravityUnit):
        self.data.tensor = unit.output_data
        # self.label.tensor = unit.output_label


class DatasetSimplePointGravityNode(WorkerNodeBase[EmptyInputs, DatasetSimplePointGravityOutputs]):
    _unit: DatasetSimplePointGravityUnit
    _params: DatasetSimplePointGravityParams

    def __init__(self, params: DatasetSimplePointGravityParams, name: str = 'DatasetSimplePointGravity',
                 seed: Optional[int] = None):
        super().__init__(name, outputs=DatasetSimplePointGravityOutputs(self))
        self._params = params.clone()
        self._seed = seed

    def _create_unit(self, creator: TensorCreator) -> Unit:
        random = get_rand_generator_or_set_cuda_seed(creator.device, self._seed)

        return DatasetSimplePointGravityUnit(creator, self._params, random)

    def _step(self):
        self._unit.step()

    @property
    def canvas_shape(self) -> List[int]:
        return list(self._params.canvas_shape)

    @canvas_shape.setter
    def canvas_shape(self, value: List[int]):
        validate_list_of_size(value, 2)
        self._params.canvas_shape = Size2D(*value)

    @property
    def point_pos(self) -> Tuple[int, int]:
        return astuple(self._params.point_pos)

    @point_pos.setter
    def point_pos(self, value: Tuple[int, int]):
        point = Point2D(*value)
        self._params.point_pos = point
        if self.is_initialized():
            self._unit.point_pos = point

    @property
    def attractor_distance(self) -> int:
        return self._params.attractor_distance

    @attractor_distance.setter
    def attractor_distance(self, value: int):
        self._params.attractor_distance = value
        if self.is_initialized():
            self._unit.attractor_distance = value

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @seed.setter
    def seed(self, value: Optional[int]):
        self._seed = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
            self._prop_builder.auto("Canvas shape", type(self).canvas_shape, edit_strategy=disable_on_runtime,
                                    hint="Size of node output [y, x]"),
            self._prop_builder.auto("Point position", type(self).point_pos, hint="Position of moving point [y, x]"),
            self._prop_builder.auto("Attractor distance", type(self).attractor_distance,
                                    hint="Distance of attractor from the moving point"),
            self._prop_builder.auto("Random seed", type(self).seed,
                                    hint="Random number generator seed, when unchecked random seed is generated")
        ]
