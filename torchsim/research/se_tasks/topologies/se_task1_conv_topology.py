import torch

from torchsim.core.actions import SpaceEngineersActionsDescriptor
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import ActionMonitorNode, AgentActionsParserNode, ExpertFlockNode, ForkNode, JoinNode, LambdaNode, \
    PassNode, RandomNumberNode, SpaceEngineersConnectorNode, SwitchNode, ToOneHotNode, ConstantNode

from torchsim.research.se_tasks.topologies.se_task_topology import TestableTopology
from torchsim.utils.space_engineers_connector import SpaceEngineersConnectorConfig


class SeT1ConvTopologicalGraph(TestableTopology):
    """Purpose of this topology is to do reserch towards solving Task 1, not solving it."""

    def __init__(self, curriculum: tuple = (1, -1)):
        super().__init__()

        se_config = SpaceEngineersConnectorConfig()
        se_config.render_width = 16
        se_config.render_height = 16
        se_config.curriculum = list(curriculum)

        base_expert_params = ExpertParams()
        base_expert_params.flock_size = 1
        base_expert_params.n_cluster_centers = 100
        base_expert_params.compute_reconstruction = False
        base_expert_params.spatial.cluster_boost_threshold = 1000
        base_expert_params.spatial.learning_rate = 0.2
        base_expert_params.spatial.batch_size = 1000
        base_expert_params.spatial.buffer_size = 1010
        base_expert_params.spatial.learning_period = 100

        base_expert_params.temporal.batch_size = 1000
        base_expert_params.temporal.buffer_size = 1010
        base_expert_params.temporal.learning_period = 200
        base_expert_params.temporal.forgetting_limit = 20000

        # parent_expert_params = ExpertParams()
        # parent_expert_params.flock_size = 1
        # parent_expert_params.n_cluster_centers = 20
        # parent_expert_params.compute_reconstruction = True
        # parent_expert_params.temporal.exploration_probability = 0.9
        # parent_expert_params.spatial.cluster_boost_threshold = 1000
        # parent_expert_params.spatial.learning_rate = 0.2
        # parent_expert_params.spatial.batch_size = 1000
        # parent_expert_params.spatial.buffer_size = 1010
        # parent_expert_params.spatial.learning_period = 100
        # parent_expert_params.temporal.context_without_rewards_size = se_config.LOCATION_SIZE_ONE_HOT

        # SE nodes
        actions_descriptor = SpaceEngineersActionsDescriptor()
        node_se_connector = SpaceEngineersConnectorNode(actions_descriptor, se_config)
        node_action_monitor = ActionMonitorNode(actions_descriptor)

        # flock-related nodes
        flock_node = ExpertFlockNode(base_expert_params)
        blank_task_control = ConstantNode((se_config.TASK_CONTROL_SIZE,))
        blank_task_labels = ConstantNode((20,))

        # parent_flock_node = ExpertFlockNode(parent_expert_params)

        join_node = JoinNode(flatten=True)

        actions = ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']
        action_count = len(actions)

        pass_actions_node = PassNode(output_shape=(action_count,), name="pass actions")
        fork_node = ForkNode(0, [base_expert_params.n_cluster_centers, action_count])

        def squeeze(inputs, outputs):
            outputs[0].copy_(inputs[0].squeeze())

        squeeze_node = LambdaNode(squeeze, 1, [(base_expert_params.n_cluster_centers + action_count,)],
                                  name="squeeze lambda node")

        def stack_and_unsqueeze(inputs, outputs):
            outputs[0].copy_(torch.stack([inputs[0], inputs[1]]).unsqueeze(0))

        stack_unsqueeze_node = LambdaNode(stack_and_unsqueeze, 2, [(1, 2, se_config.LOCATION_SIZE_ONE_HOT)],
                                          name="stack and unsqueeze node")

        to_one_hot_node = ToOneHotNode()

        action_parser_node = AgentActionsParserNode(actions_descriptor, actions)

        random_node = RandomNumberNode(0, action_count, name="random action generator", generate_new_every_n=5,
                                       randomize_intervals=True)

        switch_node = SwitchNode(2)

        # add nodes to the graph
        self.add_node(flock_node)
        # self.add_node(parent_flock_node)
        self.add_node(node_se_connector)
        self.add_node(node_action_monitor)
        self.add_node(blank_task_control)
        self.add_node(blank_task_labels)
        # self.add_node(join_node)
        # self.add_node(fork_node)
        # self.add_node(pass_actions_node)
        # self.add_node(squeeze_node)
        # self.add_node(to_one_hot_node)
        # self.add_node(stack_unsqueeze_node)
        self.add_node(action_parser_node)
        self.add_node(random_node)
        # self.add_node(switch_node)

        # first layer
        Connector.connect(
            node_se_connector.outputs.image_output,
            flock_node.inputs.sp.data_input
        )

        # Connector.connect(
        #     flock_node.outputs.tp.projection_outputs,
        #     join_node.inputs[0]
        # )
        # Connector.connect(
        #     pass_actions_node.outputs.output,
        #     join_node.inputs[1]
        # )

        # # second layer
        # Connector.connect(
        #     join_node.outputs.output,
        #     parent_flock_node.inputs.sp.data_input
        # )

        # Connector.connect(
        #     node_se_connector.outputs.task_to_agent_location_one_hot,
        #     stack_unsqueeze_node.inputs[0]
        # )
        # Connector.connect(
        #     node_se_connector.outputs.task_to_agent_location_target_one_hot,
        #     stack_unsqueeze_node.inputs[1]
        # )
        # Connector.connect(
        #     stack_unsqueeze_node.outputs[0],
        #     parent_flock_node.inputs.tp.context_input
        # )
        #
        # # actions
        # Connector.connect(
        #     parent_flock_node.outputs.sp.predicted_reconstructed_input,
        #     squeeze_node.inputs[0]
        # )
        # Connector.connect(
        #     squeeze_node.outputs[0],
        #     fork_node.inputs.input
        # )
        # Connector.connect(
        #     fork_node.outputs[1],
        #     to_one_hot_node.inputs.input
        # )
        # Connector.connect(
        #     random_node.outputs.one_hot_output,
        #     switch_node.inputs[0]
        # )
        # Connector.connect(
        #     to_one_hot_node.outputs.output,
        #     switch_node.inputs[1]
        # )
        # Connector.connect(
        #     switch_node.outputs.output,
        #     action_parser_node.inputs.input
        # )
        # directly use random exploration
        Connector.connect(
            random_node.outputs.one_hot_output,
            action_parser_node.inputs.input
        )

        Connector.connect(
            action_parser_node.outputs.output,
            node_action_monitor.inputs.action_in
        )
        # Connector.connect(
        #     switch_node.outputs.output,
        #     pass_actions_node.inputs.input,
        #     is_low_priority=True
        # )
        Connector.connect(
            node_action_monitor.outputs.action_out,
            node_se_connector.inputs.agent_action,
            # is_low_priority=True
            is_backward=False
        )

        # blank connection
        Connector.connect(blank_task_control.outputs.output,
                          node_se_connector.inputs.task_control)
        Connector.connect(blank_task_labels.outputs.output,
                          node_se_connector.inputs.agent_to_task_label)

        # Save the SE connector so we can check testing/training phase.
        # When the se_io interface has been added, this can be removed.
        self._node_se_connector = node_se_connector

    def is_in_testing_phase(self):
        if self._node_se_connector.outputs.metadata_testing_phase.tensor is None:
            return False
        else:
            return self._node_se_connector.outputs.metadata_testing_phase.tensor.cpu().item() == 1
