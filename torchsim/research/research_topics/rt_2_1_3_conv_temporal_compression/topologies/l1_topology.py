import math

from typing import Tuple

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import ConstantNode, SwitchNode
from torchsim.significant_nodes import BallEnvironment, SpReconstructionLayer, BallEnvironmentParams


class L1Topology(Topology):

    def __init__(self):
        super().__init__('cuda')

        noise_amplitude: float = 0
        env_size: Tuple[int, int] = (27, 27)
        ball_radius: int = 5
        switch_shape_after = 200

        sp_n_cluster_centers = 200  # free

        ball_env_params = BallEnvironmentParams(
            switch_shape_after=switch_shape_after,
            noise_amplitude=noise_amplitude,
            ball_radius=ball_radius,
            env_size=env_size
        )

        ball_env = BallEnvironment(ball_env_params)
        self.add_node(ball_env)
        self.ball_env = ball_env

        # topmost layer
        ep_sp = ExpertParams()
        ep_sp.flock_size = 1
        ep_sp.n_cluster_centers = sp_n_cluster_centers
        sp_reconstruction_layer = SpReconstructionLayer(env_size[0] * env_size[1],
                                                        ball_env_params.n_shapes,
                                                        sp_params=ep_sp, name="L0")
        self.add_node(sp_reconstruction_layer)
        self.sp_reconstruction_layer = sp_reconstruction_layer

        switch_node = SwitchNode(2)
        self.add_node(switch_node)
        self.switch_node = switch_node

        nan_node = ConstantNode(ball_env_params.n_shapes, math.nan)
        self.add_node(nan_node)

        Connector.connect(ball_env.outputs.data, sp_reconstruction_layer.inputs.data)

        Connector.connect(ball_env.outputs.label, switch_node.inputs[0])
        Connector.connect(nan_node.outputs.output, switch_node.inputs[1])
        Connector.connect(switch_node.outputs.output, sp_reconstruction_layer.inputs.label)

        self.is_training = True

    def restart(self):
        pass

    def switch_tt(self, train: bool):
        self.is_training = train
        self.switch_node.change_input(0 if train else 1)
        self.sp_reconstruction_layer.switch_learning(train)