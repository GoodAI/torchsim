from functools import reduce

import torch

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.models.receptive_field.grid import Stride
from torchsim.core.nodes import DatasetMNISTParams, UnsqueezeNode
from torchsim.core.nodes import DatasetSequenceMNISTNodeParams
from torchsim.core.nodes import ExpertFlockNode
from torchsim.core.nodes import JoinNode
from torchsim.core.nodes import ReceptiveFieldNode
from torchsim.core.nodes import DatasetSequenceMNISTNode
from torchsim.utils.param_utils import Size2D


class ExpertHierarchyTopology(Topology):

    def __init__(self):
        super().__init__('cuda')
        # MNIST node producing two sequences.
        # Receptive field looking at the input with window size = 14, 14, stride 7, 7.
        # 3x3 experts looking at the outputs of the RF node.
        # 1 expert combining the outputs.

        mnist_seq_params: DatasetSequenceMNISTNodeParams = DatasetSequenceMNISTNodeParams([[0, 1, 2], [3, 1, 4]])
        mnist_params = DatasetMNISTParams(class_filter=[0, 1, 2, 3, 4], one_hot_labels=False, examples_per_class=1)

        n_channels_1 = 1
        input_dims_1 = torch.Size((28, 28, n_channels_1))
        parent_rf_dims_1 = Size2D(14, 14)
        parent_rf_stride_dims_1 = Stride(7, 7)

        self.expert_params = ExpertParams()
        self.expert_params.flock_size = 9
        self.expert_params.spatial.input_size = reduce(lambda a, x: a * x, parent_rf_dims_1 + (n_channels_1,))
        self.expert_params.n_cluster_centers = 10
        self.expert_params.spatial.buffer_size = 100
        self.expert_params.spatial.batch_size = 50
        self.expert_params.spatial.learning_period = 1
        self.expert_params.spatial.cluster_boost_threshold = 100
        self.expert_params.spatial.max_boost_time = 200

        self.expert_params.temporal.seq_length = 3
        self.expert_params.temporal.seq_lookahead = 1
        self.expert_params.temporal.buffer_size = 100
        self.expert_params.temporal.batch_size = 50
        self.expert_params.temporal.learning_period = 50 + self.expert_params.temporal.seq_lookbehind
        self.expert_params.temporal.incoming_context_size = 1
        self.expert_params.temporal.max_encountered_seqs = 100
        self.expert_params.temporal.n_frequent_seqs = 20
        self.expert_params.temporal.forgetting_limit = 1000

        self.parent_expert_params = self.expert_params.clone()
        self.parent_expert_params.flock_size = 1
        self.parent_expert_params.spatial.input_size = \
            self.expert_params.flock_size * 2 * self.expert_params.n_cluster_centers

        # Create the nodes.

        mnist_node_1 = DatasetSequenceMNISTNode(params=mnist_params, seq_params=mnist_seq_params)
        mnist_node_2 = DatasetSequenceMNISTNode(params=mnist_params, seq_params=mnist_seq_params)
        receptive_field_node_1_1 = ReceptiveFieldNode(input_dims_1, parent_rf_dims_1, parent_rf_stride_dims_1)
        receptive_field_node_1_2 = ReceptiveFieldNode(input_dims_1, parent_rf_dims_1, parent_rf_stride_dims_1)
        expert_flock_node_1 = ExpertFlockNode(self.expert_params)
        expert_flock_node_2 = ExpertFlockNode(self.expert_params)
        join_node = JoinNode(dim=0, n_inputs=2)
        expert_parent_node = ExpertFlockNode(self.parent_expert_params)

        self.add_node(mnist_node_1)
        self.add_node(receptive_field_node_1_1)
        self.add_node(expert_flock_node_1)

        self.add_node(mnist_node_2)
        self.add_node(receptive_field_node_1_2)
        self.add_node(expert_flock_node_2)

        self.add_node(join_node)
        self.add_node(expert_parent_node)

        unsqueeze_node_0 = UnsqueezeNode(0)
        self.add_node(unsqueeze_node_0)

        Connector.connect(mnist_node_1.outputs.data, receptive_field_node_1_1.inputs.input)
        Connector.connect(receptive_field_node_1_1.outputs.output, expert_flock_node_1.inputs.sp.data_input)
        Connector.connect(expert_flock_node_1.outputs.tp.projection_outputs, join_node.inputs[0])

        Connector.connect(mnist_node_2.outputs.data, receptive_field_node_1_2.inputs.input)
        Connector.connect(receptive_field_node_1_2.outputs.output, expert_flock_node_2.inputs.sp.data_input)
        Connector.connect(expert_flock_node_2.outputs.tp.projection_outputs, join_node.inputs[1])

        Connector.connect(join_node.outputs.output, unsqueeze_node_0.inputs.input)
        Connector.connect(unsqueeze_node_0.outputs.output, expert_parent_node.inputs.sp.data_input)
