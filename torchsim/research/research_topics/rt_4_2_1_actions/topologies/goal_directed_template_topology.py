import numpy as np
import torch
from torchsim.core import SMALL_CONSTANT

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import NodeGroupBase, GenericGroupInputs, GenericGroupOutputs
from torchsim.core.models.expert_params import ParamsBase
from torchsim.core.nodes.grid_world_node import MultiGridWorldNode
from torchsim.core.nodes.internals.grid_world import GridWorldParams
from torchsim.core.nodes import LambdaNode, RandomNoiseNode, RandomNoiseParams, ToOneHotNode
from torchsim.core.nodes import GridWorldNode


class GoalDirectedExpertGroupInputs(GenericGroupInputs['ExpertNodeGroup']):
    def __init__(self, owner: 'GoalDirectedExpertGroupBase'):
        super().__init__(owner)

        self.data = self.create("Data")
        self.reward = self.create("Reward")


class GoalDirectedExpertGroupOutputs(GenericGroupOutputs['ExpertNodeGroup']):
    def __init__(self, owner: 'GoalDirectedExpertGroupBase'):
        super().__init__(owner)

        self.predicted_reconstructed_input = self.create("Predicted reconstructed input")


class GoalDirectedExpertGroupBase(NodeGroupBase[GoalDirectedExpertGroupInputs, GoalDirectedExpertGroupOutputs]):
    def __init__(self, name: str):
        super().__init__(name, inputs=GoalDirectedExpertGroupInputs(self), outputs=GoalDirectedExpertGroupOutputs(self))


class GoalDirectedTemplateTopologyParams(ParamsBase):
    use_egocentric: bool = False
    n_parallel_runs: int = 1
    world_params: GridWorldParams = None

    def __init__(self, use_egocentric: bool, n_parallel_runs: int, world_params: GridWorldParams):
        self.use_egocentric = use_egocentric
        self.n_parallel_runs = n_parallel_runs
        self.world_params = world_params


class GoalDirectedTemplateTopology(Topology):
    _node_grid_world: GridWorldNode

    def __init__(self, model: GoalDirectedExpertGroupBase, params: GoalDirectedTemplateTopologyParams):
        super().__init__(device='cuda')

        self.n_parallel_runs = params.n_parallel_runs

        self.worlds = MultiGridWorldNode(self.n_parallel_runs, params.world_params)

        self.model = model

        self.one_hot_node = ToOneHotNode()

        noise_params = RandomNoiseParams(amplitude=0.0001)

        self.noise_node = RandomNoiseNode(noise_params)

        def parse_action(inputs, outputs):
            probs = inputs[0]
            outputs[0].copy_(probs[:, -1, :4] + SMALL_CONSTANT)

        def shape_reward(inputs, outputs):
            rewards = inputs[0]
            rewards = torch.cat([rewards, torch.zeros_like(rewards)], dim=-1)
            outputs[0].copy_(rewards)

        self.action_parser = LambdaNode(func=parse_action, n_inputs=1, output_shapes=[(self.n_parallel_runs, 4)],
                                        name="Action parser")

        self.reward_shaper = LambdaNode(func=shape_reward, n_inputs=1, output_shapes=[(self.n_parallel_runs, 2)],
                                        name="Reward shaper")

        self.add_node(self.worlds)

        self.add_node(self.model)
        self.add_node(self.action_parser)
        self.add_node(self.reward_shaper)
        self.add_node(self.noise_node)
        self.add_node(self.one_hot_node)

        if params.use_egocentric:
            Connector.connect(self.worlds.outputs.egocentric_image_actions, self.noise_node.inputs.input)
        else:
            Connector.connect(self.worlds.outputs.output_image_actions, self.noise_node.inputs.input)

        Connector.connect(self.noise_node.outputs.output, self.model.inputs.data)

        Connector.connect(self.model.outputs.predicted_reconstructed_input, self.action_parser.inputs[0])

        Connector.connect(self.worlds.outputs.rewards, self.reward_shaper.inputs[0])
        Connector.connect(self.reward_shaper.outputs[0], self.model.inputs.reward)

        Connector.connect(self.action_parser.outputs[0], self.one_hot_node.inputs.input)

        # If a multi-action-monitor node is created, add it here between the one hot and the worlds.
        Connector.connect(self.one_hot_node.outputs.output, self.worlds.inputs.agent_actions, is_backward=True)

    def get_rewards(self) -> np.ndarray:
        if self.worlds.outputs.rewards.tensor is None:
            return np.array([0] * self.n_parallel_runs)

        return self.worlds.outputs.rewards.tensor.cpu().numpy()
