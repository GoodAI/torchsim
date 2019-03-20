import copy
from dataclasses import dataclass, field
from typing import List

import torch
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.core.physics_model.debug_world import DebugWorld
from torchsim.core.physics_model.latent_world import LatentWorld
from torchsim.core.physics_model.pymunk_physics import PyMunkPhysics, TemporalClass, Instance, InstanceColor, PymunkParams, \
    InstanceShape
from torchsim.core.physics_model.rendered_world import RenderWorld
from torchsim.gui.observables import ObserverPropertiesItem, disable_on_runtime
from torchsim.gui.validators import validate_positive_int


@dataclass
class DisentangledWorldNodeParams(ParamsBase):
    sx: int = 100
    sy: int = 100
    color_max: float = 2
    shape_max: float = 2
    max_velocity = (100., 100.)
    world_dims: [int, int] = (100, 40)
    temporal_classes: List[TemporalClass] = field(default_factory=list)
    noise_amplitude: float = 0.2
    use_pygame: bool = True
    latent_world_class = LatentWorld
    latent_world_params = {}
    render_world_class = RenderWorld
    render_world_params = {}


def create_default_temporal_classes(sx: int, sy: int):
    pmp = PymunkParams()
    pmp.sx = sx
    pmp.sy = sy

    temporal_class_definitions = \
        [TemporalClass(
            [
                Instance(pmp, 100, init_position=(20, 20), init_direction=(0, 20)),
                Instance(pmp, 101, init_position=(20, 20), init_direction=(0, 20))
            ],
        )]
    return temporal_class_definitions


class DisentangledWorldNodeUnit(Unit):
    _params: DisentangledWorldNodeParams
    bitmap: torch.Tensor
    latent: torch.Tensor

    def __init__(self, creator: TensorCreator,
                 params: DisentangledWorldNodeParams,
                 pre_collision_callback=None,
                 post_collision_callback=None):

        super().__init__(creator.device)

        self._params = copy.copy(params)
        py_params = PymunkParams()
        self._params.sx = py_params.sx
        self._params.sy = py_params.sy
        self._params.shape_max = py_params.shape_max
        self._params.color_max = py_params.color_max
        self._params.world_dims = py_params.world_dims
        self._params.max_velocity = py_params.max_velocity
        self._physics = PyMunkPhysics(params=py_params,
                                      temporal_classes=params.temporal_classes,
                                      pre_collision_callback=pre_collision_callback,
                                      post_collision_callback=post_collision_callback)

        self._latent_world = params.latent_world_class(**params.latent_world_params)

        self._debug_world = None
        if self._params.use_pygame:
            self._debug_world = DebugWorld(self._physics)

        self._physics.step()

        self.latent = self._latent_world.to_tensor(self._physics.instances)

    def step(self, *args, **kwargs):
        self._physics.step()
        self.latent.copy_(self._latent_world.to_tensor(self._physics.instances))

        if self._params.use_pygame:
            self._debug_world.show()


class DisentangledWorldNodeOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.latent = self.create("Latent vars")

    def prepare_slots(self, unit: DisentangledWorldNodeUnit):
        self.latent.tensor = unit.latent


class DisentangledWorldNode(WorkerNodeBase[EmptyInputs, DisentangledWorldNodeOutputs]):
    outputs: DisentangledWorldNodeOutputs
    pre_collision_callback = None
    post_collision_callback = None

    def __init__(self, params: DisentangledWorldNodeParams,
                 pre_collision_callback=None,
                 post_collision_callback=None,
                 name="DisentangledWorldNode"):
        super().__init__(name=name, outputs=DisentangledWorldNodeOutputs(self))

        self._params = params.clone()
        self.pre_collision_callback = pre_collision_callback
        self.post_collision_callback = post_collision_callback

    def _create_unit(self, creator: TensorCreator):
        self._creator = creator

        return DisentangledWorldNodeUnit(creator, self._params,
                                         self.pre_collision_callback,
                                         self.post_collision_callback)

    @property
    def sx(self) -> int:
        return self._params.sx

    @sx.setter
    def sx(self, value: int):
        validate_positive_int(value)
        self._params.sx = value

    @property
    def sy(self) -> int:
        return self._params.sy

    @sy.setter
    def sy(self, value: int):
        validate_positive_int(value)
        self._params.sy = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        """Define which properties can be changed from GUI and how."""

        return [
            self._prop_builder.auto('Sx', type(self).sx, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Sy', type(self).sy, edit_strategy=disable_on_runtime),
        ]

    def _step(self):
        self._unit.step()
