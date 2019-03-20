from enum import Enum
from typing import List, Tuple, Optional

import pymunk
import torch
from pymunk import Body, Vec2d
from pymunk import _chipmunk_cffi

from torchsim.core.models.expert_params import ParamsBase

cp = _chipmunk_cffi.lib
ffi = _chipmunk_cffi.ffi

from colour import Color
import numpy as np


class PymunkParams(ParamsBase):
    sx: int = 100
    sy: int = 100
    color_max: float = 2
    shape_max: float = 2
    max_velocity = (100., 100.)
    world_dims: [int, int] = (100, 40)


class Attribute(Enum):
    @classmethod
    def max(cls):
        return max(e.value for e in cls)

    def to_one_hot_tensor(self):
        t = torch.zeros(self.max())
        t[self.value] = 1
        return t

    def to_one_hot_tuple(self):
        t = [0] * self.max()
        t[self.value] = 1
        return tuple(t)

    def to_scalar(self):
        return self.value / self.max()


class InstanceShape(Attribute):
    CIRCLE = 0
    SQUARE = 1
    TRIANGLE = 2

    def to_one_hot(self) -> List[int]:
        result = [0] * (self.max() + 1)
        result[self.value] = 1
        if result == [0] * (self.max() + 1):
            raise Exception('Error!')
        return result

    @classmethod
    def from_one_hot(cls, one_hot: torch.tensor):
        if len(one_hot.nonzero()) != 1:
            raise Exception('Error!')
        else:
            return InstanceShape(int(one_hot.nonzero()))


class InstanceColor(Attribute):
    RED = 0
    GREEN = 1
    BLUE = 2

    def to_color(self):
        return Color(self.name)

    def to_one_hot(self) -> List[int]:
        result = [0] * (self.max() + 1)
        result[self.value] = 1
        if result == [0] * 3:
            raise Exception('Error!')
        return result

    @classmethod
    def from_one_hot(cls, one_hot: torch.tensor):
        if len(one_hot.nonzero()) != 1:
            raise Exception('Error!')
        else:
            return InstanceColor(int(one_hot.nonzero()))


class Instance:
    pm_body: pymunk.Body
    pm_shape: pymunk.Shape
    _params: PymunkParams

    def __init__(self,
                 params: PymunkParams,
                 instance_id: int = 0,
                 time_persistence: int = 100,
                 color: InstanceColor = InstanceColor.RED,
                 shape: InstanceShape = InstanceShape.CIRCLE,
                 init_position: Tuple = (0, 0),
                 rewrite_position: bool = False,
                 init_direction: Tuple = (0, 0),  # initial dir. of movement, will be multiplied by the object_velocity
                 rewrite_velocity: bool = True,
                 rewrite_direction: bool = False,  # change the direction during the instance switch?
                 object_velocity: int = 10):  # velocity of this object

        self.rewrite_direction = rewrite_direction
        self._params = params
        self.object_velocity = object_velocity  # how fast the object is moving
        self.time_persistence = time_persistence
        self.color = color
        self.shape = shape
        self.init_position = init_position
        self.rewrite_position = rewrite_position
        self.init_direction = self._normalize(init_direction)
        self.rewrite_velocity = rewrite_velocity
        self.instance_id = instance_id

    @staticmethod
    def _normalize(speed: (float, float)):
        """Normalize to unit length"""
        len = np.sqrt(speed[0] ** 2 + speed[1] ** 2)
        normalized = [x / len for x in speed]
        return tuple(normalized)

    @classmethod
    def from_tensor(cls, tensor: torch.tensor, params: PymunkParams):
        c_min = 4
        c_max = 4 + params.color_max + 1
        s_min = c_max
        s_max = s_min + params.shape_max + 1
        assert len(tensor) == 4 + (params.color_max + 1) + (params.shape_max + 1), 'Invalid tensor format!'

        instance = cls(params=params,
                       init_position=(tensor[0], tensor[1]),
                       init_direction=(tensor[2], tensor[3]),
                       color=InstanceColor.from_one_hot(tensor[c_min: c_max]),
                       shape=InstanceShape.from_one_hot(tensor[s_min: s_max]))

        # de-normalize position
        instance.init_position = [instance.init_position[1] * params.world_dims[1],
                                  instance.init_position[0] * params.world_dims[0]]

        return instance

    def to_tensor(self):
        return torch.tensor(
            self.pm_body.position_normalize() +  # TODO probably somewhere is a bug, second object might disappear
            self.pm_body.velocity_normalize() +
            self.color.to_one_hot() +
            self.shape.to_one_hot()
        )


class TemporalClass:
    def __init__(self, instances: List[Instance]):
        self.instances = instances


class PyMunkPhysics:
    _params: PymunkParams
    active_instances_idx: List[int]
    activation_times: List[int]
    instances: List[Instance]

    _space: pymunk.space

    ORIGIN = (0., 0.)

    def __init__(self, params: PymunkParams,
                 temporal_classes: List[TemporalClass],
                 pre_collision_callback=None,
                 post_collision_callback=None):

        self._params = params
        self.temporal_classes = temporal_classes
        self.pre_collision_callback = pre_collision_callback
        self.post_collision_callback = post_collision_callback

        self.instances = []
        self.active_instances_idx = [0] * len(temporal_classes)
        self.activation_times = [0] * len(temporal_classes)

        self._init_scene()

    def _init_scene(self):
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # Static barrier walls (lines) that the balls bounce off of
        self._add_world_bounds()
        self._add_temporal_classes()

        handler = self._space.add_default_collision_handler()
        handler.pre_solve = self.pre_collision
        handler.post_solve = self.post_collision

    def find_instance(self, arbiter: pymunk.arbiter) -> List[Instance]:
        results = []
        for instance in self.instances:
            for colider in arbiter.shapes:
                if instance.pm_shape.bb == colider.bb:
                    results.append(instance)

        return results

    def pre_collision(self, arbiter: pymunk.arbiter, space: pymunk.space, data: dict) -> bool:
        objects = self.find_instance(arbiter)

        if self.pre_collision_callback:
            self.pre_collision_callback(objects)

        return True

    def post_collision(self, arbiter: pymunk.arbiter, space: pymunk.space, data: dict) -> bool:
        objects = self.find_instance(arbiter)

        if self.post_collision_callback:
            self.post_collision_callback(objects)

        return True

    def _add_world_bounds(self):
        """
        Create the static bodies.
        :return: None
        """
        static_body = self._space.static_body
        width = self._params.world_dims[0] - self.ORIGIN[0]
        height = self._params.world_dims[1] - self.ORIGIN[1]
        static_lines = [pymunk.Segment(static_body, self.ORIGIN, (width, self.ORIGIN[1]), 0.0),
                        pymunk.Segment(static_body, (width, self.ORIGIN[1]), (width, height), 0.0),
                        pymunk.Segment(static_body, (width, height), (self.ORIGIN[0], height), 0.0),
                        pymunk.Segment(static_body, (self.ORIGIN[0], height), self.ORIGIN, 0.0)]

        for line in static_lines:
            line.elasticity = 1.0
            line.friction = 0.0

        self._space.add(static_lines)

    def _add_temporal_classes(self):
        for temporal_class in self.temporal_classes:
            first_instance = temporal_class.instances[0]
            self._add_instance(first_instance,
                               first_instance.init_position,
                               first_instance.init_direction,
                               first_instance.object_velocity)
            self.instances.append(first_instance)

    def _add_instance(self, instance: Instance, pos, direction, vel: Optional[int] = None):
        body, shape = self._create_ball(10, pos, Body.DYNAMIC)

        shape.color = [x * 255 for x in instance.color.to_color().get_rgb()] + [255]

        if vel is not None:
            direction = [vel * x for x in direction]

        body.apply_impulse_at_local_point(direction, (0, 0))

        # object reference
        instance.pm_body = body
        instance.pm_shape = shape
        instance._params = self._params
        self._space.add(body, shape)

        def normalize_position():
            return [x / m for x, m in zip(body.position.int_tuple, self._params.world_dims)]

        def denormalize_position():
            return [x * m for x, m in zip(body.position.int_tuple, self._params.world_dims)]

        def normalize_velocity():
            return list(instance._normalize(body.velocity.int_tuple))

        def denormalize_velocity():
            # not used
            return [x * m if x < m else 1 for x, m in zip(body.velocity.int_tuple, self._params.max_velocity)]

        body.position_normalize = normalize_position
        body.position_denormalize = denormalize_position
        body.velocity_normalize = normalize_velocity
        body.velocity_denormalize = denormalize_velocity

    @staticmethod
    def _create_ball(size: int, pos: [int, int], body_type: Body.body_type):
        """
        Create a ball.
        """

        mass = 1
        inertia = pymunk.moment_for_circle(mass, 0, size, (0, 0))
        body = pymunk.Body(mass, inertia, body_type)
        body.position = [pos[0], pos[1]]

        shape = pymunk.Circle(body, size, (0, 0))
        shape.elasticity = 1.0
        shape.friction = 0.0

        return body, shape

    def step(self):
        for x in range(self._physics_steps_per_frame):
            self._space.step(self._dt)
        self._update_scene()
        self.activation_times = [x + 1 for x in self.activation_times]
        self._update_temporal_classes()

    def _update_scene(self):
        pass

    def _update_temporal_classes(self):
        for i, (time, temporal_class, instance) \
                in enumerate(zip(self.activation_times, self.temporal_classes, self.instances)):
            expected_time = instance.time_persistence
            if expected_time <= time:
                instances = temporal_class.instances
                self.active_instances_idx[i] = (1 + self.active_instances_idx[i]) % len(instances)
                new_active_index = self.active_instances_idx[i]
                self.activation_times[i] = 0
                new_instance = instances[new_active_index]

                if new_instance == instance:
                    pass
                else:
                    self.replace_instance(instance, new_instance)

    def replace_instance(self, old_instance, instance):
        if instance.rewrite_position:
            pos = instance.init_position
        else:
            pos = old_instance.pm_body.position
        if instance.rewrite_direction:
            vel = [direction * instance.object_velocity for direction in instance.init_direction]
            vel = Vec2d(vel)
        else:
            if instance.rewrite_velocity:
                # change the speed according to the new instance
                old_velocity_normalized = old_instance.pm_body.velocity / old_instance.object_velocity
                vel = old_velocity_normalized * instance.object_velocity
            else:
                vel = old_instance.pm_body.velocity

        self._space.remove(old_instance.pm_body, old_instance.pm_shape)
        self._add_instance(instance, pos, vel)

        self.instances = [instance if x == old_instance else x for x in self.instances]
