import numpy as np
import torch
from torchsim.core import SMALL_CONSTANT

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes.internals.grid_world import GridWorldActionDescriptor, GridWorldParams
from torchsim.core.nodes import ActionMonitorNode, ConvExpertFlockNode, ExpertFlockNode, UnsqueezeNode, LambdaNode, \
    RandomNoiseNode, JoinNode, RandomNumberNode, SwitchNode, RandomNoiseParams, ConstantNode, ToOneHotNode
from torchsim.core.nodes import GridWorldNode
from torchsim.core.utils.tensor_utils import id_to_one_hot, normalize_probs


class GoalDirectedTopology(Topology):
    _node_grid_world: GridWorldNode
    _node_action_monitor: ActionMonitorNode

    def __init__(self):
        super().__init__(device='cuda')
        actions_descriptor = GridWorldActionDescriptor()
        node_action_monitor = ActionMonitorNode(actions_descriptor)

        params = GridWorldParams(map_name='MapE')
        noise_params = RandomNoiseParams(amplitude=0.0001)
        node_grid_world = GridWorldNode(params)
        expert_params = ExpertParams()
        unsqueeze_node = UnsqueezeNode(dim=0)
        noise_node = RandomNoiseNode(noise_params)
        constant_node = ConstantNode(shape=(1, 1, 3, 48))
        one_hot_node = ToOneHotNode()

        def context(inputs, outputs):
            con = inputs[0]
            con[:, :, 1:, 24:] = float('nan')
            outputs[0].copy_(con)

        def f(inputs, outputs):
            probs = inputs[0]
            outputs[0].copy_(probs[0, -1, :4] + SMALL_CONSTANT)

        action_parser = LambdaNode(func=f, n_inputs=1, output_shapes=[(4,)])
        context_assembler = LambdaNode(func=context, n_inputs=1, output_shapes=[(1, 1, 3, 48)])

        expert_params.flock_size = 1
        expert_params.n_cluster_centers = 24
        expert_params.produce_actions = True
        expert_params.temporal.seq_length = 9
        expert_params.temporal.seq_lookahead = 7
        expert_params.temporal.n_frequent_seqs = 700
        expert_params.temporal.max_encountered_seqs = 1000
        expert_params.temporal.exploration_probability = 0.01
        expert_params.temporal.batch_size = 200
        expert_params.temporal.own_rewards_weight = 20
        expert_params.temporal.incoming_context_size = 48

        expert_params.compute_reconstruction = True

        #expert_node = ConvExpertFlockNode(expert_params)
        expert_node = ExpertFlockNode(expert_params)

        self.add_node(node_grid_world)
        self.add_node(node_action_monitor)
        self.add_node(expert_node)
        self.add_node(unsqueeze_node)
        self.add_node(action_parser)
        self.add_node(noise_node)
        self.add_node(constant_node)
        self.add_node(context_assembler)
        self.add_node(one_hot_node)

        Connector.connect(node_grid_world.outputs.egocentric_image_action, noise_node.inputs.input)
        Connector.connect(noise_node.outputs.output, unsqueeze_node.inputs.input)
        Connector.connect(unsqueeze_node.outputs.output, expert_node.inputs.sp.data_input)
        Connector.connect(node_grid_world.outputs.reward, expert_node.inputs.tp.reward_input)

        Connector.connect(constant_node.outputs.output, context_assembler.inputs[0])
        Connector.connect(context_assembler.outputs[0], expert_node.inputs.tp.context_input)

        Connector.connect(expert_node.outputs.sp.predicted_reconstructed_input, action_parser.inputs[0])
        Connector.connect(action_parser.outputs[0], one_hot_node.inputs.input)
        Connector.connect(one_hot_node.outputs.output, node_action_monitor.inputs.action_in)
        Connector.connect(node_action_monitor.outputs.action_out, node_grid_world.inputs.agent_action, is_backward=True)


