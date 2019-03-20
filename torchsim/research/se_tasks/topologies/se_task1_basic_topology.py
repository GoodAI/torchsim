import logging

import torch

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import NodeGroupBase, GroupInputs, GroupOutputs
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import ExpertFlockNode, ForkNode, JoinNode, LambdaNode, UnsqueezeNode
from torchsim.core.nodes.grayscale_node import GrayscaleNode
from torchsim.research.se_tasks.topologies.se_task_topology import TestableTopology
from torchsim.research.se_tasks.topologies.task1_base_topology import Task1BaseGroup, Task1BaseGroupWorld

logger = logging.getLogger(__name__)


class Task1BasicExpertGroupInputs(GroupInputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.actions = self.create("Actions")
        self.image = self.create("Image")
        self.current_location = self.create("Current location")
        self.target_location = self.create("Target location")


class Task1BasicExpertGroupOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.actions = self.create("Actions")


class Task1BasicExpertGroup(NodeGroupBase[Task1BasicExpertGroupInputs, Task1BasicExpertGroupOutputs]):
    """
    A model which receives images, target locations and current locations from the 1th SE task.
    """
    RANDOM_INPUT_ID: int = 0
    ARCHITECTURE_INPUT_ID: int = 1

    def __init__(self, action_count=4, location_vector_size=100, use_grayscale: bool = False):
        super().__init__("Task 1 - Basic expert", inputs=Task1BasicExpertGroupInputs(self),
                         outputs=Task1BasicExpertGroupOutputs(self))

        base_expert_params = ExpertParams()
        base_expert_params.flock_size = 1
        base_expert_params.n_cluster_centers = 100
        base_expert_params.compute_reconstruction = False
        base_expert_params.spatial.cluster_boost_threshold = 1000
        base_expert_params.spatial.learning_rate = 0.2
        base_expert_params.spatial.batch_size = 1000
        base_expert_params.spatial.buffer_size = 1010
        base_expert_params.spatial.learning_period = 100

        parent_expert_params = ExpertParams()
        parent_expert_params.flock_size = 1
        parent_expert_params.n_cluster_centers = 20
        parent_expert_params.compute_reconstruction = True
        parent_expert_params.temporal.exploration_probability = 0.9
        parent_expert_params.spatial.cluster_boost_threshold = 1000
        parent_expert_params.spatial.learning_rate = 0.2
        parent_expert_params.spatial.batch_size = 1000
        parent_expert_params.spatial.buffer_size = 1010
        parent_expert_params.spatial.learning_period = 100
        parent_expert_params.temporal.context_without_rewards_size = location_vector_size

        # flock-related nodes
        flock_node = ExpertFlockNode(base_expert_params)

        parent_flock_node = ExpertFlockNode(parent_expert_params)

        join_node = JoinNode(flatten=True)

        unsqueeze_node_to_base_expert = UnsqueezeNode(0)

        unsqueeze_node_to_parent_expert = UnsqueezeNode(0)

        fork_node = ForkNode(0, [base_expert_params.n_cluster_centers, action_count])

        def squeeze(inputs, outputs):
            outputs[0].copy_(inputs[0].squeeze())

        squeeze_node = LambdaNode(squeeze, 1, [(base_expert_params.n_cluster_centers + action_count,)],
                                  name="squeeze lambda node")

        def stack_and_unsqueeze(inputs, outputs):
            outputs[0].copy_(torch.stack([inputs[0], inputs[1]]).unsqueeze(0))

        stack_unsqueeze_node = LambdaNode(stack_and_unsqueeze, 2, [(1, 2, location_vector_size)],
                                          name="stack and unsqueeze node")

        # add nodes to the graph
        self.add_node(flock_node)
        self.add_node(unsqueeze_node_to_base_expert)
        self.add_node(parent_flock_node)
        self.add_node(unsqueeze_node_to_parent_expert)
        self.add_node(join_node)
        self.add_node(fork_node)
        self.add_node(squeeze_node)
        self.add_node(stack_unsqueeze_node)

        Connector.connect(
            self.inputs.actions.output,
            join_node.inputs[1]
        )

        if use_grayscale:
            grayscale_node = GrayscaleNode(squeeze_channel=True)
            self.add_node(grayscale_node)
            Connector.connect(
                self.inputs.image.output,
                grayscale_node.inputs.input
            )
            Connector.connect(
                grayscale_node.outputs.output,
                unsqueeze_node_to_base_expert.inputs.input
            )
        else:
            Connector.connect(
                self.inputs.image.output,
                unsqueeze_node_to_base_expert.inputs.input
            )

        Connector.connect(
            unsqueeze_node_to_base_expert.outputs.output,
            flock_node.inputs.sp.data_input
        )

        Connector.connect(
            self.inputs.current_location.output,
            stack_unsqueeze_node.inputs[0]
        )

        Connector.connect(
            self.inputs.target_location.output,
            stack_unsqueeze_node.inputs[1]
        )

        Connector.connect(
            fork_node.outputs[1],
            self.outputs.actions.input
        )

        # first layer
        Connector.connect(
            flock_node.outputs.tp.projection_outputs,
            join_node.inputs[0]
        )

        # second layer
        Connector.connect(
            join_node.outputs.output,
            unsqueeze_node_to_parent_expert.inputs.input
        )

        # second layer
        Connector.connect(
            unsqueeze_node_to_parent_expert.outputs.output,
            parent_flock_node.inputs.sp.data_input
        )

        Connector.connect(
            stack_unsqueeze_node.outputs[0],
            parent_flock_node.inputs.tp.context_input
        )

        # actions
        Connector.connect(
            parent_flock_node.outputs.sp.predicted_reconstructed_input,
            squeeze_node.inputs[0]
        )
        Connector.connect(
            squeeze_node.outputs[0],
            fork_node.inputs.input
        )

        # Save the SE connector so we can check testing/training phase.
        # When the se_io interface has been added, this can be removed.


class SeT1Bt(TestableTopology):
    """Purpose of this topology is to do research towards solving Task 1, not solving it."""

    def __init__(self, action_count=4, location_vector_size=100, curriculum: tuple = (1, -1)):
        super().__init__()
        base = Task1BaseGroup()
        base_world = Task1BaseGroupWorld(curriculum=curriculum)
        expert_topology = Task1BasicExpertGroup(action_count=action_count,
                                                location_vector_size=location_vector_size,
                                                use_grayscale=False)

        self.add_node(base)
        self.add_node(base_world)
        self.add_node(expert_topology)

        base.outputs.connect_automatically(base_world.inputs, is_backward=True)
        base_world.outputs.connect_automatically(expert_topology.inputs)
        expert_topology.outputs.connect_automatically(base.inputs)

        self.node_se_connector = base_world.node_se_connector
