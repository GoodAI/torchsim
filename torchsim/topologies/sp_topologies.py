from torchsim.core.nodes import ConvSpatialPoolerFlockNode, TemporalPoolerFlockNode
from torchsim.core.nodes import DatasetMNISTNode, DatasetMNISTParams
from torchsim.core.nodes import ExpandNode
from torchsim.core.nodes import JoinNode
from torchsim.core.nodes import SpatialPoolerFlockNode
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams


class SpatialPoolerTopology(Topology):
    sp_params: ExpertParams
    mnist_params: DatasetMNISTParams

    def __init__(self):
        super().__init__(device='cuda')

        self.sp_params = ExpertParams()
        self.sp_params.n_cluster_centers = 4
        self.sp_params.spatial.input_size = 28 * 28
        self.sp_params.flock_size = 3
        self.sp_params.spatial.buffer_size = 100  # TODO (Bug): setting the buffer size around 300+ causes the model to crash
        self.sp_params.spatial.batch_size = 45

        self.mnist_params = DatasetMNISTParams(class_filter=[0, 1, 2, 3, 4], one_hot_labels=False)

        # self.sp_params.custom_input_shape = [self.sp_params.flock_size, 28,
        #                                      28]  # has to be here (flock_size can change)

        spatial_pooler = SpatialPoolerFlockNode(self.sp_params.clone())
        expand = ExpandNode(dim=0, desired_size=self.sp_params.flock_size)
        mnist = DatasetMNISTNode(params=self.mnist_params)

        self.add_node(mnist)
        self.add_node(spatial_pooler)
        self.add_node(expand)

        Connector.connect(mnist.outputs.data, expand.inputs.input)
        Connector.connect(expand.outputs.output, spatial_pooler.inputs.sp.data_input)


class SpatialPoolerHierarchy(Topology):
    sp_params1: ExpertParams
    sp_params_parent: ExpertParams
    # TODO (?): isn't it dangerous to initialize variables here (they are then shared among instances)?
    mnist_params: DatasetMNISTParams = DatasetMNISTParams([0, 1, 2, 3, 1, 4])

    def __init__(self):
        super().__init__(device='cuda')

        self.sp_params1 = ExpertParams()
        self.sp_params1.n_cluster_centers = 4
        self.sp_params1.spatial.input_size = 28 * 28
        self.sp_params1.flock_size = 3

        self.sp_params2 = ExpertParams()
        self.sp_params2.n_cluster_centers = 3
        self.sp_params2.spatial.input_size = 28 * 28
        self.sp_params2.flock_size = 3

        self.sp_params_parent = ExpertParams()
        self.sp_params_parent.n_cluster_centers = 2
        self.sp_params_parent.spatial.input_size = self.sp_params1.n_cluster_centers + self.sp_params2.n_cluster_centers
        self.sp_params_parent.flock_size = 3

        _node_mnist = DatasetMNISTNode(params=self.mnist_params)

        _node_sp1 = SpatialPoolerFlockNode(self.sp_params1)
        expand1 = ExpandNode(dim=0,
                             desired_size=self.sp_params1.flock_size)

        expand2 = ExpandNode(dim=0,
                             desired_size=self.sp_params2.flock_size)

        _node_sp2 = SpatialPoolerFlockNode(self.sp_params2)
        _node_sp_parent = SpatialPoolerFlockNode(self.sp_params_parent)
        _join = JoinNode(dim=1, n_inputs=2)

        self.add_node(_node_mnist)
        self.add_node(_node_sp1)
        self.add_node(expand1)
        self.add_node(expand2)
        self.add_node(_node_sp2)
        self.add_node(_node_sp_parent)
        self.add_node(_join)

        Connector.connect(_node_mnist.outputs.data, expand1.inputs.input)
        Connector.connect(expand1.outputs.output, _node_sp1.inputs.sp.data_input)

        Connector.connect(_node_mnist.outputs.data, expand2.inputs.input)
        Connector.connect(expand2.outputs.output, _node_sp2.inputs.sp.data_input)

        Connector.connect(_node_sp1.outputs.sp.forward_clusters, _join.inputs[0])
        Connector.connect(_node_sp2.outputs.sp.forward_clusters, _join.inputs[1])

        Connector.connect(_join.outputs.output, _node_sp_parent.inputs.sp.data_input)


class ConvSpatialPoolerTopology(Topology):
    sp_params: ExpertParams
    mnist_params: DatasetMNISTParams

    def __init__(self):
        super().__init__(device='cuda')

        self.sp_params = ExpertParams()
        self.sp_params.n_cluster_centers = 4
        self.sp_params.spatial.input_size = 28 * 28
        self.sp_params.flock_size = 4
        self.sp_params.spatial.buffer_size = 100
        self.sp_params.spatial.batch_size = 45

        mnist_params0 = DatasetMNISTParams(class_filter=[0], one_hot_labels=False)
        mnist_params1 = DatasetMNISTParams(class_filter=[1], one_hot_labels=False)
        mnist_params2 = DatasetMNISTParams(class_filter=[2], one_hot_labels=False)
        mnist_params3 = DatasetMNISTParams(class_filter=[3], one_hot_labels=False)

        spatial_pooler = ConvSpatialPoolerFlockNode(self.sp_params.clone())
        mnist0 = DatasetMNISTNode(params=mnist_params0)
        mnist1 = DatasetMNISTNode(params=mnist_params1)
        mnist2 = DatasetMNISTNode(params=mnist_params2)
        mnist3 = DatasetMNISTNode(params=mnist_params3)

        expand0 = ExpandNode(dim=0, desired_size=1)
        expand1 = ExpandNode(dim=0, desired_size=1)
        expand2 = ExpandNode(dim=0, desired_size=1)
        expand3 = ExpandNode(dim=0, desired_size=1)

        join = JoinNode(dim=0, n_inputs=4)

        self.add_node(mnist0)
        self.add_node(mnist1)
        self.add_node(mnist2)
        self.add_node(mnist3)

        self.add_node(expand0)
        self.add_node(expand1)
        self.add_node(expand2)
        self.add_node(expand3)

        self.add_node(spatial_pooler)
        self.add_node(join)

        Connector.connect(mnist0.outputs.data, expand0.inputs.input)
        Connector.connect(mnist1.outputs.data, expand1.inputs.input)
        Connector.connect(mnist2.outputs.data, expand2.inputs.input)
        Connector.connect(mnist3.outputs.data, expand3.inputs.input)

        Connector.connect(expand0.outputs.output, join.inputs[0])
        Connector.connect(expand1.outputs.output, join.inputs[1])
        Connector.connect(expand2.outputs.output, join.inputs[2])
        Connector.connect(expand3.outputs.output, join.inputs[3])

        Connector.connect(join.outputs.output, spatial_pooler.inputs.sp.data_input)


class SpatialTemporalPoolerTopology(Topology):
    params: ExpertParams
    mnist_params: DatasetMNISTParams

    def __init__(self):
        super().__init__(device='cuda')

        self.params = ExpertParams()
        self.params.flock_size = 1
        self.params.spatial.input_size = 28 * 28
        self.params.n_cluster_centers = 5
        self.params.spatial.buffer_size = 100
        self.params.spatial.batch_size = 50
        self.params.spatial.learning_period = 10
        self.params.spatial.cluster_boost_threshold = 100

        self.params.temporal.seq_length = 3
        self.params.temporal.seq_lookahead = 1
        self.params.temporal.buffer_size = 100
        self.params.temporal.batch_size = 50
        self.params.temporal.learning_period = 50 + self.params.temporal.seq_lookbehind
        self.params.temporal.incoming_context_size = 1
        self.params.temporal.max_encountered_seqs = 100
        self.params.temporal.n_frequent_seqs = 20
        self.params.temporal.forgetting_limit = 1000




        # self.params = ExpertParams()
        # self.params.n_cluster_centers = 4
        # self.params.spatial.input_size = 28 * 28
        # self.params.flock_size = 3
        # self.params.spatial.buffer_size = 100  # TODO (Bug): setting the buffer size around 300+ causes the model to crash
        # self.params.spatial.batch_size = 45

        self.mnist_params = DatasetMNISTParams(class_filter=[0, 1, 2, 3, 4], one_hot_labels=False)

        # self.sp_params.custom_input_shape = [self.sp_params.flock_size, 28,
        #                                      28]  # has to be here (flock_size can change)

        spatial_pooler = SpatialPoolerFlockNode(self.params.clone())
        temporal_pooler = TemporalPoolerFlockNode(self.params.clone())
        expand = ExpandNode(dim=0, desired_size=self.params.flock_size)
        mnist = DatasetMNISTNode(params=self.mnist_params)

        self.add_node(mnist)
        self.add_node(spatial_pooler)
        self.add_node(temporal_pooler)
        self.add_node(expand)

        Connector.connect(mnist.outputs.data, expand.inputs.input)
        Connector.connect(expand.outputs.output, spatial_pooler.inputs.sp.data_input)
        Connector.connect(spatial_pooler.outputs.sp.forward_clusters, temporal_pooler.inputs.tp.data_input)
