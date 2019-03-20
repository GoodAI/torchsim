from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.models.receptive_field.grid import Stride
from torchsim.core.nodes.dataset_simple_point_gravity_node import DatasetSimplePointGravityNode, \
    DatasetSimplePointGravityParams, MoveStrategy
from torchsim.utils.param_utils import Size2D, Point2D
from torchsim.core.nodes import ConvExpertFlockNode
from torchsim.core.nodes import ReceptiveFieldNode
from torchsim.core.nodes.receptive_field_reverse_node import ReceptiveFieldReverseNode


class ReceptiveFieldTopology(Topology):
    """
    ReceptiveFieldReverseNode demonstration topology.
    """

    def __init__(self):
        super().__init__('cuda')

        expert1_cluster_centers = 6
        expert2_cluster_centers = 10  # 50
        n_channels_1 = 1
        rf1_input_size = (6, 8, n_channels_1)
        rf1_rf_size = Size2D(2, 2)
        rf2_input_size = (3, 4, expert1_cluster_centers)
        rf2_rf_size = Size2D(2, 2)
        rf2_stride = Stride(1, 1)

        self.expert_params = ExpertParams()
        self.expert_params.flock_size = 3 * 4
        self.expert_params.n_cluster_centers = expert1_cluster_centers
        self.expert_params.spatial.buffer_size = 100
        self.expert_params.spatial.batch_size = 50
        self.expert_params.spatial.learning_period = 1
        self.expert_params.spatial.cluster_boost_threshold = 100
        self.expert_params.spatial.max_boost_time = 200

        self.expert_params.temporal.seq_length = 3
        self.expert_params.temporal.seq_lookahead = 1
        self.expert_params.temporal.buffer_size = 100  # 1000
        self.expert_params.temporal.batch_size = 50  # 500
        self.expert_params.temporal.learning_period = 50 + self.expert_params.temporal.seq_lookbehind
        self.expert_params.temporal.max_encountered_seqs = 100  # 2000
        self.expert_params.temporal.n_frequent_seqs = 20
        self.expert_params.temporal.forgetting_limit = 1000  # 10000

        self.expert_params_2 = self.expert_params.clone()
        self.expert_params_2.flock_size = 2 * 3
        self.expert_params_2.n_cluster_centers = expert2_cluster_centers
        self.expert_params_2.temporal.incoming_context_size = 1
        # self.expert_params_2.temporal.max_encountered_seqs = 2000
        # self.expert_params_2.temporal.n_frequent_seqs = 500
        # self.expert_params_2.temporal.forgetting_limit = 5000
        # self.expert_params_2.temporal.context_without_rewards_size = 80

        self.expert_params.temporal.n_providers = rf2_rf_size[0] * rf2_rf_size[1] * 2
        self.expert_params.temporal.incoming_context_size = self.expert_params_2.n_cluster_centers

        # context_input_dims = (2, self.expert_params.temporal.context_without_rewards_size * 2)
        context_input_dims = (*rf2_input_size[:2], *rf2_rf_size, 2, 3, expert2_cluster_centers)

        # Create the nodes.
        receptive_field_1_node = ReceptiveFieldNode(rf1_input_size, rf1_rf_size, Stride(*rf1_rf_size))
        receptive_field_2_node = ReceptiveFieldNode(rf2_input_size, rf2_rf_size, rf2_stride)

        receptive_field_reverse_node = ReceptiveFieldReverseNode(context_input_dims, rf2_input_size, rf2_rf_size,
                                                                 rf2_stride)
        expert_flock_1_node = ConvExpertFlockNode(self.expert_params)
        expert_flock_2_node = ConvExpertFlockNode(self.expert_params_2)

        dataset_simple_point_gravity_node = DatasetSimplePointGravityNode(DatasetSimplePointGravityParams(
            canvas_shape=Size2D(6, 8),
            point_pos=Point2D(2, 3),
            attractor_distance=2,
            move_strategy=MoveStrategy.LIMITED_TO_LEFT_DOWN_QUADRANT
        ))

        self.add_node(dataset_simple_point_gravity_node)
        self.add_node(receptive_field_1_node)
        self.add_node(expert_flock_1_node)
        self.add_node(receptive_field_2_node)
        self.add_node(expert_flock_2_node)

        self.add_node(receptive_field_reverse_node)

        Connector.connect(dataset_simple_point_gravity_node.outputs.data, receptive_field_1_node.inputs.input)
        Connector.connect(receptive_field_1_node.outputs.output, expert_flock_1_node.inputs.sp.data_input)
        Connector.connect(expert_flock_1_node.outputs.tp.projection_outputs, receptive_field_2_node.inputs.input)
        Connector.connect(receptive_field_2_node.outputs.output, expert_flock_2_node.inputs.sp.data_input)
        Connector.connect(expert_flock_2_node.outputs.output_context, receptive_field_reverse_node.inputs.input)

        Connector.connect(receptive_field_reverse_node.outputs.output, expert_flock_1_node.inputs.tp.context_input,
                          is_backward=True)
