from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes import SimpleBouncingBallNode
from torchsim.core.nodes.simple_bouncing_ball_node import SimpleBouncingBallNodeParams, BallShapes


class TestSimpleBouncingBallNode:

    def test_node(self):
        sx: int = 15
        sy: int = 37
        ball_radius: int = 7
        noise_amplitude: float = 0.3
        switch_shape_after: int = 99
        random_position_direction_switch_after: int = 154

        shapes = [
            BallShapes.CIRCLE,
            BallShapes.EMPTY_TRIANGLE
        ]

        creator = AllocatingCreator(device="cpu")

        bouncing_params = SimpleBouncingBallNodeParams(sx=sx,
                                                       sy=sy,
                                                       ball_radius=ball_radius,
                                                       ball_shapes=shapes,
                                                       dir_x=2,
                                                       dir_y=1,
                                                       noise_amplitude=noise_amplitude,
                                                       switch_next_shape_after=switch_shape_after,
                                                       random_position_direction_switch_after=
                                                       random_position_direction_switch_after,
                                                       )

        node = SimpleBouncingBallNode(bouncing_params)
        node.allocate_memory_blocks(creator)
        node.validate()

        assert node.ball_shapes == [1, 5], "Incorrect ball shapes!"

        assert node.ball_radius == ball_radius, "Incorrect ball radius!"

        assert node.sx == sx, "Incorrect parameter!"

        assert node.sy == sy, "Incorrect parameter!"

        assert node.noise_amplitude == noise_amplitude, "Incorrect parameter!"

        assert node.switch_next_shape_after == switch_shape_after, "Incorrect parameter!"

        assert node.random_position_direction_switch_after == random_position_direction_switch_after,\
            "Incorrect parameter!"

        num_steps = 5
        for i in range(num_steps):
            node._step()

