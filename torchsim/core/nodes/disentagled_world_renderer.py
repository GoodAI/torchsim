import copy
from typing import List

import torch

from torchsim.core.graph.unit import Unit
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.graph.slot_container import Inputs, MemoryBlocks
from torchsim.core.nodes.disentangled_world_node import DisentangledWorldNodeParams
from torchsim.core.physics_model.pymunk_physics import PymunkParams


class DisentangledWorldRendererUnit(Unit):
    bitmap: torch.Tensor
    _params: DisentangledWorldNodeParams

    def __init__(self, creator: TensorCreator, params: DisentangledWorldNodeParams):
        super().__init__(creator.device)

        self._params = copy.copy(params)

        self._render_world = params.render_world_class((params.sx, params.sy), **params.render_world_params)

        width = self._render_world.world_size[0]
        height = self._render_world.world_size[1]
        self.bitmap = creator.zeros((width, height, 3),
                                     dtype=self._float_dtype,
                                     device=self._device)

    def step(self, instances: torch.Tensor):
        params = PymunkParams()
        self._params.sx = params.sx
        self._params.sy = params.sy
        self._params.shape_max = params.shape_max
        self._params.color_max = params.color_max
        self._params.world_dims = params.world_dims
        self._params.max_velocity = params.max_velocity
        self.bitmap.copy_(self._render_world.to_tensor(instances, params))


class DisentangledWorldRendererInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.latent = self.create("Latent vars")


class DisentangledWorldRendererOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.bitmap = self.create("Bitmap")

    def prepare_slots(self, unit: DisentangledWorldRendererUnit):
        self.bitmap.tensor = unit.bitmap


class DisentangledWorldRendererNode(WorkerNodeBase[DisentangledWorldRendererInputs, DisentangledWorldRendererOutputs]):

    def __init__(self, params: DisentangledWorldNodeParams, name="DisentangledWorldRenderer"):
        super().__init__(name=name,
                         inputs=DisentangledWorldRendererInputs(self),
                         outputs=DisentangledWorldRendererOutputs(self))

        self._params = params.clone()

    def _create_unit(self, creator: TensorCreator) -> Unit:
        self._creator = creator

        return DisentangledWorldRendererUnit(creator, self._params)

    def _step(self):
        self._unit.step(self.inputs.latent.tensor)
