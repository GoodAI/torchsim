import copy
import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

import torch
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.gui.observables import disable_on_runtime
from torchsim.gui.observer_system import ObserverPropertiesItem
from torchsim.gui.validators import *

logger = logging.getLogger(__name__)


class SimpleBall:
    """Simulates the ball physics."""
    _sx: int
    _sy: int
    _ball_radius: int
    _pos: np.array
    _direction: np.array

    def __init__(self, sx: int, sy: int, dir_x=1, dir_y=1, ball_radius=0) -> None:
        self._sx = sx
        self._sy = sy
        self._ball_radius = ball_radius

        if self._ball_radius * 2 + 1 > self._sx or self._ball_radius * 2 + 1 > self._sy:
            print('error: the ball has to fit in the bitmap (ball_radius * 2 +1 <= sx; sy)')

        # in case the ball just fits the width/height, disable movement in the corresponding direction
        if self._sx == self._ball_radius * 2 + 1:
            dir_x = 0
        if self._sy == self._ball_radius * 2 + 1:
            dir_y = 0

        self._pos = self._random_position()
        self._direction = np.array([dir_y, dir_x])  # dimensions are in this order because of rendering

    def _random_position(self):
        return np.array(
            [np.random.randint(0+self._ball_radius, self._sy-self._ball_radius),
             np.random.randint(0+self._ball_radius, self._sx-self._ball_radius)])

    def next_frame(self):
        """Simulates the ball movement, resolves bounces, direction."""
        pos = self._pos + self._direction
        self._pos[0], self._direction[0] = self._bounce(self._direction[0], pos[0], self._sy)  # y
        self._pos[1], self._direction[1] = self._bounce(self._direction[1], pos[1], self._sx)  # x

    def _bounce(self, direction: int, value: int, dim_size: int) -> [int, int]:
        if value < self._ball_radius:
            new_direction = -direction
            return (self._ball_radius + 1), new_direction

        if value >= dim_size - self._ball_radius:
            new_direction = -direction
            return (dim_size - self._ball_radius - 2), new_direction

        return value, direction

    def get_pos(self):
        return self._pos


class BallShapes(Enum):
    DISC = 0
    CIRCLE = 1
    SQUARE = 2
    EMPTY_SQUARE = 3
    TRIANGLE = 4
    EMPTY_TRIANGLE = 5

    def __str__(self):
        return self._name_

    __repr__ = __str__


class BallRenderer:
    """Renders the ball of a given shape on a given position."""
    _ball_radius: int
    _ball_shape: BallShapes

    def __init__(self, ball_radius: int, ball_shape: BallShapes):
        self._ball_radius = ball_radius
        self._ball_shape = ball_shape
        self.shape_indices = {}

    def render_ball_to(self, pos: np.array, bitmap: torch.Tensor):
        if self._ball_shape not in self.shape_indices:
            if self._ball_shape == BallShapes.CIRCLE:
                indices = self._render_circle_ball_to(self._ball_radius)
            elif self._ball_shape == BallShapes.DISC:
                indices = self._render_disc_ball_to(self._ball_radius)
            elif self._ball_shape == BallShapes.SQUARE:
                indices = self._render_square_ball_to(self._ball_radius)
            elif self._ball_shape == BallShapes.EMPTY_SQUARE:
                indices = self._render_empty_square_ball_to(self._ball_radius)
            elif self._ball_shape == BallShapes.EMPTY_TRIANGLE:
                indices = self._render_empty_triangle_ball_to(self._ball_radius)
            elif self._ball_shape == BallShapes.TRIANGLE:
                indices = self._render_triangle_ball_to(self._ball_radius)
            else:
                raise ValueError("Unknown shape.")

            self.shape_indices[self._ball_shape] = np.array(list(zip(*indices)))

        indices = self.shape_indices[self._ball_shape] + np.expand_dims(pos, 1)
        indices = indices[:, (0 < indices[0, :]) * (indices[0, :] < bitmap.shape[0]) *
                             (0 < indices[1, :]) * (indices[1, :] < bitmap.shape[1])]

        bitmap[indices] = 1

    @staticmethod
    def _render_square_ball_to(radius):
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                yield y, x

    @staticmethod
    def _render_empty_square_ball_to(radius):
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                if BallRenderer._is_point_on_boundary(np.array([0, 0]), np.array([y, x]), radius):
                    yield y, x

    @staticmethod
    def _render_empty_triangle_ball_to(radius):
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                if BallRenderer.is_point_on_triangle(np.array([0, 0]), np.array([y, x]), radius):
                    yield y, x

    @staticmethod
    def _render_triangle_ball_to(radius):
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                if BallRenderer.is_point_inside_triangle(np.array([0, 0]), np.array([y, x]), radius):
                    yield y, x

    @staticmethod
    def _render_disc_ball_to(radius):
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                if BallRenderer._euclidean_dist(np.array([0, 0]), np.array([y, x])) <= radius+0.3:
                    yield y, x

    @staticmethod
    def _render_circle_ball_to(radius):
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                if radius-0.49 \
                        <= BallRenderer._euclidean_dist(np.array([0, 0]), np.array([y, x])) \
                        <= radius+0.49:
                    yield y, x

    @staticmethod
    def _is_point_on_boundary(a: np.array, b: np.array, radius: int):
        dx = abs(a[1] - b[1])
        dy = abs(a[0] - b[0])
        return dx == radius or dy == radius

    @staticmethod
    def _euclidean_dist(a: np.array, b: np.array):
        dy = a[0] - b[0]
        dx = a[1] - b[1]
        return np.math.sqrt(dy * dy + dx * dx)

    @staticmethod
    def is_point_on_triangle(a: np.array, b: np.array, radius):
        dy = a[0] - b[0]
        dx = a[1] - b[1]
        if dy == radius // 2:
            return True
        if abs(dx) - dy == np.ceil(radius / 2):
            return True

    @staticmethod
    def is_point_inside_triangle(a: np.array, b: np.array, radius):
        dy = a[0] - b[0]
        dx = a[1] - b[1]
        if abs(dx) - dy <= radius // 2 and dy <= radius // 2:
            return True


@dataclass
class SimpleBouncingBallNodeParams(ParamsBase):
    sx: int = 27
    sy: int = 40
    dir_x: int = 1
    dir_y: int = 1
    ball_radius: int = 1
    ball_shapes: List[BallShapes] = field(default_factory=list)
    noise_amplitude: float = 0.2
    switch_next_shape_after: int = 0
    random_position_direction_switch_after: int = 0


class SimpleBouncingBallUnit(Unit):
    """A world containing 1-pixel ball which can move in 8 direction and bounces from the walls."""
    _params: SimpleBouncingBallNodeParams
    _noise_amplitude: float
    _bitmap: torch.Tensor

    _ball: SimpleBall
    _ball_renderer: BallRenderer

    def __init__(self, creator: TensorCreator, params: SimpleBouncingBallNodeParams):
        super().__init__(creator.device)

        self._params = copy.copy(params)
        if len(self._params.ball_shapes) == 0:
            self._params.ball_shapes = [shape for shape in BallShapes]

        self._step_shape_switch_counter = 0
        self._step_direction_switch_counter = 0
        self._shape_counter = 0
        self._creator = creator

        self._ball = SimpleBall(self._params.sx,
                                self._params.sy,
                                self._params.dir_x,
                                self._params.dir_y,
                                self._params.ball_radius)

        self._ball_renderer = BallRenderer(ball_radius=self._params.ball_radius,
                                           ball_shape=self._params.ball_shapes[0])

        # size_y, size_x, 1 color channel
        self._bitmap = self._creator.zeros((self._params.sy, self._params.sx, 1),
                                           dtype=self._float_dtype,
                                           device=self._device)

        self._label = self._creator.zeros(len(self._params.ball_shapes),
                                          dtype=self._float_dtype,
                                          device=self._device)
        self._label[0] = 1

    def step(self):

        background = self._creator.zeros_like(self._bitmap)
        background = background.uniform_() * self._params.noise_amplitude

        self._switch_shape()

        self._switch_position_and_direction()

        self._ball.next_frame()
        self._ball_renderer.render_ball_to(self._ball.get_pos(), background)

        self._bitmap.copy_(background)

    def _switch_shape(self):
        if self._params.switch_next_shape_after > 0:
            self._step_shape_switch_counter += 1
            if self._step_shape_switch_counter % self._params.switch_next_shape_after == 0:
                self._shape_counter += 1
                self._ball_shape = self._params.ball_shapes[self._shape_counter % len(self._params.ball_shapes)]
                self._ball_renderer = BallRenderer(ball_radius=self._params.ball_radius,
                                                   ball_shape=self._ball_shape)
                self._label.zero_()
                self._label[self._shape_counter % len(self._params.ball_shapes)] = 1

    def _switch_position_and_direction(self):
        if self._params.random_position_direction_switch_after > 0:

            self._step_direction_switch_counter += 1
            if self._step_direction_switch_counter % self._params.random_position_direction_switch_after == 0:
                rand_dir_x = rand_dir_y = 0
                # repeat until we have a nonzero speed in both directions
                while rand_dir_x == 0 or rand_dir_y == 0:
                    rand_dir_x, rand_dir_y = np.random.randint(-2, 3), np.random.randint(-2, 3)
                self._ball = SimpleBall(self._params.sx, self._params.sy, rand_dir_x, rand_dir_y, self._params.ball_radius)


class SimpleBouncingBallOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.bitmap = self.create("Bitmap")
        self.label_one_hot = self.create("Label one hot")

    def prepare_slots(self, unit: SimpleBouncingBallUnit):
        self.bitmap.tensor = unit._bitmap
        self.label_one_hot.tensor = unit._label


class SimpleBouncingBallNode(WorkerNodeBase[EmptyInputs, SimpleBouncingBallOutputs]):
    """Simple world containing one moving object at a time and noise.

    Size, direction of movement, size of object, switch period between two shapes
    and amplitude of noise can be specified.
    """

    outputs: SimpleBouncingBallOutputs

    def __init__(self, params: SimpleBouncingBallNodeParams, name="SimpleBouncingBallWorld"):
        super().__init__(name=name, outputs=SimpleBouncingBallOutputs(self))

        self._params = params.clone()

    def _create_unit(self, creator: TensorCreator):

        self._creator = creator

        return SimpleBouncingBallUnit(creator, self._params)

    @property
    def ball_shapes(self) ->List[int]:
        return [item.value for item in self._params.ball_shapes]

    @ball_shapes.setter
    def ball_shapes(self, value: List[int]):
        parsed_value = [BallShapes(item) for item in value]

        if len(parsed_value) == 0:
            raise FailedValidationException(f"Value must not be empty")

        self._params.ball_shapes = parsed_value

    @property
    def ball_radius(self) -> int:
        return self._params.ball_radius

    @ball_radius.setter
    def ball_radius(self, value: int):
        validate_positive_int(value)
        if int(value) * 2 + 1 > self._params.sx or int(value) * 2 + 1 > self._params.sy:
            raise FailedValidationException('The ball (of size 2*ball_diameter+1) has to fit inside the [sx, sy] dimensions')
        self._params.ball_radius = value

    @property
    def switch_next_shape_after(self) -> int:
        return self._params.switch_next_shape_after

    @switch_next_shape_after.setter
    def switch_next_shape_after(self, value: int):
        validate_positive_int(value)
        self._params.switch_next_shape_after = value

    @property
    def sx(self) -> int:
        return self._params.sx

    @sx.setter
    def sx(self, value: int):
        validate_positive_int(value)
        if int(value) < self._params.ball_radius * 2 + 1:
            raise FailedValidationException("The ball (of size 2*ball_radius+1) has to fit inside [sx, sy] dimensions")
        self._params.sx = value

    @property
    def sy(self) -> int:
        return self._params.sy

    @sy.setter
    def sy(self, value: int):
        validate_positive_int(value)
        if int(value) < self._params.ball_radius * 2 + 1:
            raise FailedValidationException("The ball (of size 2*ball_radius+1) has to fit inside [sx, sy] dimensions")
        self._params.sy = value

    @property
    def noise_amplitude(self) -> float:
        return self._params.noise_amplitude

    @noise_amplitude.setter
    def noise_amplitude(self, value: float):
        validate_positive_with_zero_float(value)
        self._params.noise_amplitude = value

    @property
    def dir_x(self) -> int:
        return self._params.dir_x

    @dir_x.setter
    def dir_x(self, value: int):
        if value not in [0, 1, 2]:
            raise FailedValidationException("Invalid direction, allowed values are [0,1,2] (1 is no movement)")
        self._params.dir_x = value

    @property
    def dir_y(self) -> int:
        return self._params.dir_y

    @dir_y.setter
    def dir_y(self, value: int):
        if value not in [0, 1, 2]:
            raise FailedValidationException("Invalid direction, allowed values are [0,1,2] (1 is no movement)")
        self._params.dir_y = value

    @property
    def random_position_direction_switch_after(self) -> int:
        return self._params.random_position_direction_switch_after

    @random_position_direction_switch_after.setter
    def random_position_direction_switch_after(self, value: int):
        validate_positive_with_zero_int(value)
        self._params.random_position_direction_switch_after = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        """Define which properties can be changed from GUI and how."""

        return [
            self._prop_builder.auto('Random reset', type(self).random_position_direction_switch_after, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Sx', type(self).sx, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Sy', type(self).sy, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Dir_x', type(self).dir_x, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Dir_y', type(self).dir_y, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Ball radius', type(self).ball_radius, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Ball shapes', type(self).ball_shapes, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Noise amplitude', type(self).noise_amplitude, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Switch next shape after', type(self).switch_next_shape_after, edit_strategy=disable_on_runtime)
            ]

    def _step(self):
        self._unit.step()
