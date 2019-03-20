from typing import Tuple, List

import torch
from torchsim.core.nodes import BallShapes
from torchsim.core.nodes.simple_bouncing_ball_node import BallRenderer
from torchsim.core.physics_model.pymunk_physics import InstanceShape, Instance, PymunkParams


class RenderWorld:
    def __init__(self, world_size: Tuple[int, int], radius: int = 9):
        self.world_size = world_size
        self.shapes_renderer_mapping = {InstanceShape.CIRCLE: BallRenderer(radius, BallShapes.DISC),
                                        InstanceShape.SQUARE: BallRenderer(radius, BallShapes.SQUARE),
                                        InstanceShape.TRIANGLE: BallRenderer(radius, BallShapes.TRIANGLE)}

    def to_tensor(self, tensors: List[torch.Tensor], params: PymunkParams):
        scene = torch.zeros(tuple(self.world_size) + (3,))
        for tensor in tensors:
            instance = Instance.from_tensor(tensor, params)
            channel = torch.zeros(tuple(self.world_size))
            self.shapes_renderer_mapping[instance.shape].render_ball_to(instance.init_position, channel)
            rgb = instance.color.to_color().get_rgb()
            for c in range(3):
                scene[:, :, c] += channel * rgb[c]
        return scene
