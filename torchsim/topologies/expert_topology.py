from torchsim.core.nodes import DatasetMNISTParams, DatasetSequenceMNISTNodeParams, DatasetSequenceMNISTNode
from torchsim.core.nodes import ExpertFlockNode
from torchsim.core.graph.connection import Connector
from torchsim.core.graph import Topology
from torchsim.core.models.expert_params import ExpertParams
import numpy as np
from torchsim.core.nodes import UnsqueezeNode


class ExpertTopology(Topology):

    def __init__(self):
        super().__init__(device='cuda')
        # One MNIST producing two sequences, and one ExpertFlock learning to recognize them

        self.expert_params = ExpertParams()
        self.expert_params.flock_size = 1
        self.expert_params.spatial.input_size = 28 * 28
        self.expert_params.n_cluster_centers = 5
        self.expert_params.spatial.buffer_size = 100
        self.expert_params.spatial.batch_size = 50
        self.expert_params.spatial.learning_period = 10
        self.expert_params.spatial.cluster_boost_threshold = 100

        self.expert_params.temporal.seq_length = 3
        self.expert_params.temporal.seq_lookahead = 1
        self.expert_params.temporal.buffer_size = 100
        self.expert_params.temporal.batch_size = 50
        self.expert_params.temporal.learning_period = 50 + self.expert_params.temporal.seq_lookbehind
        self.expert_params.temporal.incoming_context_size = 1
        self.expert_params.temporal.max_encountered_seqs = 100
        self.expert_params.temporal.n_frequent_seqs = 20
        self.expert_params.temporal.forgetting_limit = 1000

        mnist_seq_params = DatasetSequenceMNISTNodeParams([[0, 1, 2], [3, 1, 4]], np.array([[0.9, 0.1], [0.9, 0.1]]))
        mnist_params = DatasetMNISTParams(class_filter=[0, 1, 2, 3, 4], one_hot_labels=False, examples_per_class=1)

        mnist_seq_node = DatasetSequenceMNISTNode(params=mnist_params, seq_params=mnist_seq_params, seed=5)
        # mnist_node = DatasetMNISTNode(params=mnist_params, seed=5)
        expert_node = ExpertFlockNode(self.expert_params, seed=2)

        # self.add_node(mnist_node)
        self.add_node(mnist_seq_node)
        self.add_node(expert_node)

        unsqueeze_node_0 = UnsqueezeNode(0)
        self.add_node(unsqueeze_node_0)

        Connector.connect(mnist_seq_node.outputs.data, unsqueeze_node_0.inputs.input)
        Connector.connect(unsqueeze_node_0.outputs.output, expert_node.inputs.sp.data_input)



