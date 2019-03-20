import numpy as np
import torch
from torchsim.core import SMALL_CONSTANT

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes.internals.grid_world import GridWorldActionDescriptor, GridWorldParams
from torchsim.core.nodes import ActionMonitorNode, ExpertFlockNode, UnsqueezeNode, LambdaNode, \
    RandomNoiseNode, RandomNoiseParams, ToOneHotNode
from torchsim.core.nodes import GridWorldNode
from torchsim.core.utils.tensor_utils import id_to_one_hot, normalize_probs


class GoalDirectedNarrowHierarchyTopology(Topology):
    _node_grid_world: GridWorldNode
    _node_action_monitor: ActionMonitorNode

    def __init__(self):
        super().__init__(device='cuda')
        actions_descriptor = GridWorldActionDescriptor()
        node_action_monitor = ActionMonitorNode(actions_descriptor)

        params = GridWorldParams(map_name='MapE')
        noise_params = RandomNoiseParams(amplitude=0.0001)
        node_grid_world = GridWorldNode(params)
        expert_params1 = ExpertParams()
        unsqueeze_node = UnsqueezeNode(dim=0)
        noise_node = RandomNoiseNode(noise_params)
        one_hot_node = ToOneHotNode()

        def f(inputs, outputs):
            probs = inputs[0]
            outputs[0].copy_(probs[0, -1, :4] + SMALL_CONSTANT)

        action_parser = LambdaNode(func=f, n_inputs=1, output_shapes=[(4,)])

        expert_params1.flock_size = 1
        expert_params1.n_cluster_centers = 24
        expert_params1.produce_actions = True
        expert_params1.temporal.seq_length = 4
        expert_params1.temporal.seq_lookahead = 2
        expert_params1.temporal.n_frequent_seqs = 700
        expert_params1.temporal.max_encountered_seqs = 1000
        expert_params1.temporal.exploration_probability = 0.05
        expert_params1.temporal.batch_size = 200
        expert_params1.temporal.frustration_threshold = 2
        # expert_params.temporal.own_rewards_weight = 20

        expert_params1.compute_reconstruction = True

        expert_params2 = expert_params1.clone()
        expert_params2.temporal.seq_length = 5
        expert_params2.temporal.seq_lookahead = 4
        expert_params2.n_cluster_centers = 8
        expert_params2.produce_actions = False
        expert_params2.temporal.frustration_threshold = 10

        #expert_params1.temporal.incoming_context_size = 2 * expert_params2.n_cluster_centers

        expert_node1 = ExpertFlockNode(expert_params1)
        expert_node2 = ExpertFlockNode(expert_params2)

        self.add_node(node_grid_world)
        self.add_node(node_action_monitor)
        self.add_node(expert_node1)
        self.add_node(expert_node2)
        self.add_node(unsqueeze_node)
        self.add_node(action_parser)
        self.add_node(noise_node)
        self.add_node(one_hot_node)

        Connector.connect(node_grid_world.outputs.output_image_action, noise_node.inputs.input)
        Connector.connect(noise_node.outputs.output, unsqueeze_node.inputs.input)
        Connector.connect(unsqueeze_node.outputs.output, expert_node1.inputs.sp.data_input)

        Connector.connect(expert_node1.outputs.tp.projection_outputs, expert_node2.inputs.sp.data_input)
        Connector.connect(expert_node2.outputs.output_context, expert_node1.inputs.tp.context_input, is_backward=True)

        Connector.connect(expert_node1.outputs.sp.predicted_reconstructed_input, action_parser.inputs[0])

        Connector.connect(node_grid_world.outputs.reward, expert_node1.inputs.tp.reward_input)
        Connector.connect(node_grid_world.outputs.reward, expert_node2.inputs.tp.reward_input)

        Connector.connect(action_parser.outputs[0], one_hot_node.inputs.input)
        Connector.connect(one_hot_node.outputs.output, node_action_monitor.inputs.action_in)

        Connector.connect(node_action_monitor.outputs.action_out, node_grid_world.inputs.agent_action, is_backward=True)


