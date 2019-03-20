import numpy as np
import torch
from torchsim.core import SMALL_CONSTANT

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes.internals.grid_world import GridWorldActionDescriptor, GridWorldParams, ResetStrategy
from torchsim.core.nodes import ActionMonitorNode, ConvExpertFlockNode, ExpertFlockNode, UnsqueezeNode, LambdaNode, \
    RandomNoiseNode, JoinNode, RandomNumberNode, SwitchNode, RandomNoiseParams, ConstantNode, ToOneHotNode
from torchsim.core.nodes import GridWorldNode
from torchsim.core.utils.tensor_utils import id_to_one_hot, normalize_probs


class SingleExpertExperimentalTopology(Topology):
    _node_grid_world: GridWorldNode
    _node_action_monitor: ActionMonitorNode

    def __init__(self):
        super().__init__(device='cuda')
        actions_descriptor = GridWorldActionDescriptor()
        node_action_monitor = ActionMonitorNode(actions_descriptor)

        params = GridWorldParams(map_name='MapTwoRoom', reset_strategy=ResetStrategy.ANYWHERE)
        noise_params = RandomNoiseParams(amplitude=0.0001)
        node_grid_world = GridWorldNode(params)
        expert_params = ExpertParams()
        unsqueeze_node = UnsqueezeNode(dim=0)
        noise_node = RandomNoiseNode(noise_params)
        one_hot_node = ToOneHotNode()

        def f(inputs, outputs):
            probs = inputs[0]
            outputs[0].copy_(probs[0, -1, :4] + SMALL_CONSTANT)

        action_parser = LambdaNode(func=f, n_inputs=1, output_shapes=[(4,)])

        expert_params.flock_size = 1
        expert_params.n_cluster_centers = 64
        expert_params.produce_actions = True
        expert_params.temporal.seq_length = 17
        expert_params.temporal.seq_lookahead = 13
        expert_params.temporal.n_frequent_seqs = 700
        expert_params.temporal.max_encountered_seqs = 1000
        expert_params.temporal.exploration_probability = 0.05
        expert_params.temporal.batch_size = 200
        expert_params.temporal.buffer_size = 1000
        expert_params.temporal.own_rewards_weight = 20
        expert_params.temporal.frustration_threshold = 2
        expert_params.temporal.compute_backward_pass = True

        expert_params.compute_reconstruction = True

        expert_node = ConvExpertFlockNode(expert_params)
        #expert_node = ExpertFlockNode(expert_params)

        self.add_node(node_grid_world)
        self.add_node(node_action_monitor)
        self.add_node(expert_node)
        self.add_node(unsqueeze_node)
        self.add_node(action_parser)
        self.add_node(noise_node)
        self.add_node(one_hot_node)

        Connector.connect(node_grid_world.outputs.egocentric_image_action, noise_node.inputs.input)
        Connector.connect(noise_node.outputs.output, unsqueeze_node.inputs.input)
        Connector.connect(unsqueeze_node.outputs.output, expert_node.inputs.sp.data_input)
        Connector.connect(node_grid_world.outputs.reward, expert_node.inputs.tp.reward_input)

        Connector.connect(expert_node.outputs.sp.predicted_reconstructed_input, action_parser.inputs[0])
        Connector.connect(action_parser.outputs[0], one_hot_node.inputs.input)
        Connector.connect(one_hot_node.outputs.output, node_action_monitor.inputs.action_in)
        Connector.connect(node_action_monitor.outputs.action_out, node_grid_world.inputs.agent_action, is_backward=True)


