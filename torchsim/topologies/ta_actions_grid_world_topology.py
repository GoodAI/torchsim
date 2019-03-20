from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes.internals.grid_world import GridWorldActionDescriptor, GridWorldParams
from torchsim.core.nodes import ActionMonitorNode, UnsqueezeNode
from torchsim.core.nodes import ExpertFlockNode
from torchsim.core.nodes import ForkNode
from torchsim.core.nodes import GridWorldNode
from torchsim.core.nodes import JoinNode
from torchsim.core.nodes import LambdaNode
from torchsim.core.nodes import RandomNumberNode


class TaActionsGridWorldTopology(Topology):
    _node_grid_world: GridWorldNode
    _node_action_monitor: ActionMonitorNode

    def __init__(self):
        super().__init__("cuda")
        actions_descriptor = GridWorldActionDescriptor()
        node_action_monitor = ActionMonitorNode(actions_descriptor)

        grid_world_params = GridWorldParams('MapE')
        grid_world_params.tile_size = 3
        node_grid_world = GridWorldNode(grid_world_params)

        random_action_generator = RandomNumberNode(upper_bound=len(actions_descriptor.action_names()))

        join_node = JoinNode(flatten=True)

        # GridWorld sizes
        width = grid_world_params.egocentric_width * grid_world_params.tile_size
        height = grid_world_params.egocentric_height * grid_world_params.tile_size
        fork_node = ForkNode(dim=0, split_sizes=[width * height, 4])

        self.add_node(node_grid_world)
        self.add_node(node_action_monitor)
        self.add_node(random_action_generator)
        self.add_node(join_node)
        self.add_node(fork_node)

        Connector.connect(node_grid_world.outputs.egocentric_image, join_node.inputs[0])
        Connector.connect(node_grid_world.outputs.output_action, join_node.inputs[1])

        self._create_and_connect_agent(join_node, fork_node)

        Connector.connect(random_action_generator.outputs.one_hot_output,
                          node_action_monitor.inputs.action_in)
        Connector.connect(node_action_monitor.outputs.action_out,
                          node_grid_world.inputs.agent_action)
        # Connector.connect(fork_node.outputs[1], node_grid_world.inputs.agent_action, is_low_priority=True)

    def _create_and_connect_agent(self, join_node: JoinNode, fork_node: ForkNode):
        params = ExpertParams()
        params.flock_size = 1
        params.n_cluster_centers = 28
        params.compute_reconstruction = True
        params.spatial.cluster_boost_threshold = 1000
        params.spatial.buffer_size = 500
        params.spatial.batch_size = 500
        params.spatial.learning_rate = 0.3
        params.spatial.learning_period = 50

        # conv_expert = ConvExpertFlockNode(params, name="Conv. expert")
        # conv_expert = SpatialPoolerFlockNode(params, name=" SP")
        conv_expert = ExpertFlockNode(params, name=" expert")

        self.add_node(conv_expert)

        unsqueeze_node_0 = UnsqueezeNode(0)
        self.add_node(unsqueeze_node_0)

        Connector.connect(join_node.outputs.output, unsqueeze_node_0.inputs.input)
        Connector.connect(unsqueeze_node_0.outputs.output, conv_expert.inputs.sp.data_input)

        def squeeze(inputs, outputs):
            outputs[0].copy_(inputs[0].squeeze(0))

        squeeze_node = LambdaNode(squeeze, 1, [(sum(fork_node._split_sizes),)],
                                  name="squeeze lambda node")

        self.add_node(squeeze_node)

        Connector.connect(conv_expert.outputs.sp.predicted_reconstructed_input, squeeze_node.inputs[0])
        Connector.connect(squeeze_node.outputs[0], fork_node.inputs.input)
