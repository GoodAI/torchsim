import random
from typing import List

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.nodes import RandomNoiseNode, RandomNoiseParams
from torchsim.core.nodes.disentagled_world_renderer import DisentangledWorldRendererNode
from torchsim.core.nodes.disentangled_world_node import DisentangledWorldNodeParams, DisentangledWorldNode
from torchsim.core.physics_model.pymunk_physics import TemporalClass, Instance, InstanceColor, InstanceShape, PymunkParams


class DisentangledWorldNodeTopology(Topology):

    def __init__(self):
        super().__init__(device='cpu')

        sx = 40
        sy = 100

        pymunk_params = PymunkParams()
        pymunk_params.sx = sx
        pymunk_params.sy = sy

        temporal_classes = \
            [
                TemporalClass([
                    Instance(params=pymunk_params,
                             time_persistence=100,
                             init_position=(20, 20),
                             init_direction=(1, 0),
                             object_velocity=110)
                ]),
                TemporalClass([
                    Instance(params=pymunk_params,
                             time_persistence=100,
                             init_position=(40, 20),
                             init_direction=(-100, -10),
                             color=InstanceColor.GREEN,
                             shape=InstanceShape.SQUARE,
                             object_velocity=30),

                    Instance(params=pymunk_params,
                             time_persistence=200,
                             init_position=(40, 30),
                             init_direction=(1, 0.1),
                             color=InstanceColor.BLUE,
                             shape=InstanceShape.TRIANGLE,
                             object_velocity=100),

                    Instance(params=pymunk_params,
                             time_persistence=50,
                             init_position=(40, 30),
                             init_direction=(1, 0.1),
                             color=InstanceColor.RED,
                             shape=InstanceShape.CIRCLE,
                             object_velocity=100)
                ])
            ]

        params = DisentangledWorldNodeParams(sx=sx, sy=sy,
                                             temporal_classes=temporal_classes,
                                             use_pygame=False)

        disentagled_world_node = DisentangledWorldNode(params=params,
                                                       post_collision_callback=None)
        disentagled_world_renderer = DisentangledWorldRendererNode(params=params)

        noise = RandomNoiseNode(RandomNoiseParams(amplitude=0.0))  # TODO support also non-zero noise amplitude

        self.add_node(disentagled_world_node)
        self.add_node(disentagled_world_renderer)
        self.add_node(noise)

        Connector.connect(disentagled_world_node.outputs.latent, noise.inputs.input)
        Connector.connect(noise.outputs.output, disentagled_world_renderer.inputs.latent)

    # gives the list of colliding objects in the parameter
    # allows for any change of colliding objects
    def post_collision_random_colors(self, instances: List[Instance]):
        """An example of the callback function, changes color of all collided objects to different color"""

        for inst in instances:
            color = inst.color
            new_color = color

            while new_color == color:
                new_color = random.choice(list(InstanceColor))

            inst.color = new_color
