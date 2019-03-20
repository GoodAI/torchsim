from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams, NUMBER_OF_CONTEXT_TYPES
from torchsim.core.nodes import DatasetMNISTParams, DatasetSequenceMNISTNodeParams, DatasetSequenceMNISTNode
from torchsim.core.nodes import ExpertFlockNode
from torchsim.core.nodes import ConstantNode
from torchsim.core.nodes import UnsqueezeNode


class LoopingTopology(Topology):
    def __init__(self):
        super().__init__(device='cuda')

        params1 = ExpertParams()
        params1.flock_size = 1
        params1.n_cluster_centers = 10
        params1.spatial.buffer_size = 100
        params1.temporal.buffer_size = 100
        params1.temporal.incoming_context_size = 9
        params1.temporal.n_providers = 2
        params1.spatial.batch_size = 50
        params1.temporal.batch_size = 50
        params1.spatial.input_size = 28 * 28

        params2 = ExpertParams()
        params2.flock_size = 1
        params2.n_cluster_centers = 9
        params2.spatial.buffer_size = 100
        params2.temporal.buffer_size = 100
        params2.temporal.incoming_context_size = 5
        params2.temporal.n_providers = 2
        params2.spatial.batch_size = 50
        params2.temporal.batch_size = 50
        params2.spatial.input_size = params1.n_cluster_centers

        mnist_seq_params = DatasetSequenceMNISTNodeParams([[0, 1, 2], [3, 1, 4]])
        mnist_params = DatasetMNISTParams(class_filter=[0, 1, 2, 3, 4], one_hot_labels=False, examples_per_class=1)
        mnist_node = DatasetSequenceMNISTNode(params=mnist_params, seq_params=mnist_seq_params)

        zero_context = ConstantNode(
            shape=(params2.flock_size, params2.temporal.n_providers, NUMBER_OF_CONTEXT_TYPES, params2.temporal.incoming_context_size),
            constant=0)

        node1 = ExpertFlockNode(params1)
        node2 = ExpertFlockNode(params2)

        self.add_node(mnist_node)
        self.add_node(node1)
        self.add_node(node2)
        self.add_node(zero_context)

        unsqueeze_node_0 = UnsqueezeNode(0)
        self.add_node(unsqueeze_node_0)

        Connector.connect(mnist_node.outputs.data, unsqueeze_node_0.inputs.input)
        Connector.connect(unsqueeze_node_0.outputs.output, node1.inputs.sp.data_input)
        Connector.connect(node1.outputs.tp.projection_outputs, node2.inputs.sp.data_input)
        Connector.connect(node2.outputs.output_context, node1.inputs.tp.context_input, is_backward=True)
        Connector.connect(zero_context.outputs.output, node2.inputs.tp.context_input)
