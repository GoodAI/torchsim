import math

from typing import Tuple, List

from torchsim.core.graph.connection import Connector
from torchsim.core.nodes import SimpleBouncingBallNode, BallShapes, SwitchNode, ConstantNode
from torchsim.core.nodes.simple_bouncing_ball_node import SimpleBouncingBallNodeParams, validate_predicate
from torchsim.significant_nodes.environment_base import EnvironmentParamsBase, EnvironmentBase


class BallEnvironmentParams(EnvironmentParamsBase):
    noise_amplitude: float = 0
    env_size: Tuple[int, int, int] = (24, 24, 1)
    ball_radius: int = 5
    switch_shape_after: int = 200
    random_position_direction_switch_after: int = 200
    shapes: List[BallShapes] = None
    n_shapes: int = -1

    DEFAULT_SHAPES = [
        BallShapes.CIRCLE,
        BallShapes.EMPTY_SQUARE,
        BallShapes.EMPTY_TRIANGLE
    ]

    def __init__(self, noise_amplitude: float = 0, env_size: Tuple[int, int] = (24, 24), ball_radius: int = 5,
                 switch_shape_after: int = 200, shapes: List[BallShapes] = None,
                 random_position_direction_switch_after=200):
        self.noise_amplitude = noise_amplitude
        self.env_size = env_size + (1,)
        self.ball_radius = ball_radius
        self.switch_shape_after = switch_shape_after
        self.random_position_direction_switch_after = random_position_direction_switch_after
        if shapes is None:
            self.shapes = self.DEFAULT_SHAPES
        else:
            self.shapes = shapes
        self.n_shapes = len(self.shapes)


class BallEnvironment(EnvironmentBase):
    def __init__(self, params: BallEnvironmentParams, name: str = "BallEnvironment"):
        super().__init__(params, name)

        bouncing_params = SimpleBouncingBallNodeParams(sx=params.env_size[0],
                                                       sy=params.env_size[1],
                                                       ball_radius=params.ball_radius,
                                                       ball_shapes=params.shapes,
                                                       dir_x=1,
                                                       dir_y=2,
                                                       noise_amplitude=params.noise_amplitude,
                                                       switch_next_shape_after=params.switch_shape_after,
                                                       random_position_direction_switch_after=
                                                       params.random_position_direction_switch_after
                                                       )

        ball_node = SimpleBouncingBallNode(bouncing_params)

        self.add_node(ball_node)
        self.ball_node = ball_node

        Connector.connect(ball_node.outputs.bitmap, self.outputs.data.input)

        switch_node = SwitchNode(2)
        self.add_node(switch_node)
        self.switch_node = switch_node

        nan_node = ConstantNode(params.n_shapes, math.nan)
        self.add_node(nan_node)

        Connector.connect(ball_node.outputs.label_one_hot, switch_node.inputs[0])
        Connector.connect(nan_node.outputs.output, switch_node.inputs[1])
        Connector.connect(switch_node.outputs.output, self.outputs.label.input)

    def switch_learning(self, on):
        self.switch_node.change_input(0 if on else 1)

    def get_correct_label_memory_block(self):
        return self.ball_node.outputs.label_one_hot

    @staticmethod
    def validate_params(params: BallEnvironmentParams):
        validate_predicate(lambda: params.noise_amplitude >= 0)
        validate_predicate(lambda: len(params.env_size) == 3 and params.env_size[0] >= 1 and params.env_size[1] >= 1
                                   and params.env_size[2] == 1)
        validate_predicate(lambda: params.ball_radius >= 1)
        validate_predicate(lambda: params.switch_shape_after >= 0)
        validate_predicate(lambda: params.random_position_direction_switch_after >= 0)
        validate_predicate(lambda: params.n_shapes > 0)
        validate_predicate(lambda: len(params.shapes) == params.n_shapes)
