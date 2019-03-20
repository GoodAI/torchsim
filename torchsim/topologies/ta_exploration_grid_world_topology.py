from typing import Tuple

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slots import MemoryBlock, InputSlot
from torchsim.core.nodes import UnsqueezeNode, RandomNumberNode
from torchsim.core.nodes.actions_monitor_node import ActionMonitorNode
from torchsim.core.nodes.fork_node import ForkNode
from torchsim.core.nodes.grid_world_node import GridWorldNode
from torchsim.core.nodes.internals.grid_world import GridWorldActionDescriptor, GridWorldParams
from torchsim.core.nodes.join_node import JoinNode
from torchsim.core.nodes.lambda_node import LambdaNode
from torchsim.core.nodes.random_noise_node import RandomNoiseNode, RandomNoiseParams
from torchsim.core.nodes.to_one_hot_node import ToOneHotNode
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.significant_nodes import ConvLayer
from torchsim.topologies.toyarch_groups.r1ncm_group import R1NCMGroup


class TaExplorationGridWorldTopology(Topology):
    """Topology for testing exploration mechanisms in TA, currently just 1 expert."""
    _node_grid_world: GridWorldNode
    _node_action_monitor: ActionMonitorNode

    def __init__(self):
        super().__init__("cuda")
        actions_descriptor = GridWorldActionDescriptor()
        node_action_monitor = ActionMonitorNode(actions_descriptor)

        grid_world_params = GridWorldParams('MapE')
        grid_world_params.tile_size = 5
        node_grid_world = GridWorldNode(grid_world_params)

        join_node = JoinNode(flatten=True)

        unsqueeze_node = UnsqueezeNode(dim=0)

        # GridWorld sizes
        # egocentric
        width = grid_world_params.egocentric_width * grid_world_params.tile_size
        height = grid_world_params.egocentric_height * grid_world_params.tile_size

        #one-hot matrix
        width = grid_world_params.world_width
        height = grid_world_params.world_height

        fork_node = ForkNode(dim=0, split_sizes=[width * height, 4])
        input_size = (1, width * height + 4)
        random_noise_node_params = RandomNoiseParams()
        random_noise_node_params.amplitude = 0.1
        random_noise_node = RandomNoiseNode(random_noise_node_params)

        def squeeze(inputs, outputs):
            outputs[0].copy_(inputs[0].view(-1))

        squeeze_node = LambdaNode(squeeze, 1, [(sum(fork_node._split_sizes),)],
                                  name="squeeze lambda node")

        to_one_hot_node = ToOneHotNode()

        random_action_generator = RandomNumberNode(upper_bound=4)

        self.add_node(squeeze_node)
        self.add_node(node_grid_world)
        self.add_node(unsqueeze_node)
        self.add_node(node_action_monitor)
        self.add_node(join_node)
        self.add_node(fork_node)
        self.add_node(random_noise_node)
        self.add_node(to_one_hot_node)
        self.add_node(random_action_generator)

        Connector.connect(node_grid_world.outputs.egocentric_image, random_noise_node.inputs.input)
        # egocentric
        # Connector.connect(random_noise_node.outputs.output, join_node.inputs[0])
        # one-hot matrix
        Connector.connect(node_grid_world.outputs.output_pos_one_hot_matrix, join_node.inputs[0])
        Connector.connect(node_grid_world.outputs.output_action, join_node.inputs[1])
        Connector.connect(join_node.outputs.output, unsqueeze_node.inputs.input)

        self._create_and_connect_agent(unsqueeze_node.outputs.output, squeeze_node.inputs[0], input_size + (1,))

        Connector.connect(squeeze_node.outputs[0], fork_node.inputs.input)

        # Connector.connect(fork_node.outputs[1],
        #                   to_one_hot_node.inputs.input)
        Connector.connect(random_action_generator.outputs.one_hot_output, to_one_hot_node.inputs.input)

        Connector.connect(to_one_hot_node.outputs.output,
                          node_action_monitor.inputs.action_in)
        Connector.connect(node_action_monitor.outputs.action_out,
                          node_grid_world.inputs.agent_action, is_backward=True)

    def _create_and_connect_agent(self, input_image: MemoryBlock, output_reconstruction: InputSlot,
                                  input_size: Tuple[int, int, int]):

        params = MultipleLayersParams()
        params.num_conv_layers = 3
        params.n_cluster_centers = [28, 14, 7]
        params.compute_reconstruction = True
        params.conv_classes = ConvLayer
        params.sp_buffer_size = 5000
        params.sp_batch_size = 500
        params.learning_rate = 0.2
        params.cluster_boost_threshold = 1000
        params.max_encountered_seqs = 1000
        params.max_frequent_seqs = 500
        params.seq_lookahead = 2
        params.seq_length = 4
        params.exploration_probability = 0
        params.rf_size = [(input_size[0], input_size[1]), (1, 1), (1, 1)]
        params.rf_stride = None
        ta_group = R1NCMGroup(conv_layers_params=params, model_seed=None, image_size=input_size)

        self.add_node(ta_group)

        Connector.connect(
            input_image,
            ta_group.inputs.image
        )

        Connector.connect(
            ta_group.outputs.predicted_reconstructed_input,
            output_reconstruction
        )




