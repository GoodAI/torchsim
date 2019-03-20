from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import ConvExpertFlockNode
from torchsim.core.nodes import DatasetMNISTParams, DatasetMNISTNode
from torchsim.core.nodes import ExpandNode
from torchsim.core.nodes import JoinNode


class ConvExpertTopology(Topology):
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

        expert = ConvExpertFlockNode(self.sp_params.clone())
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

        self.add_node(expert)
        self.add_node(join)

        Connector.connect(mnist0.outputs.data, expand0.inputs.input)
        Connector.connect(mnist1.outputs.data, expand1.inputs.input)
        Connector.connect(mnist2.outputs.data, expand2.inputs.input)
        Connector.connect(mnist3.outputs.data, expand3.inputs.input)

        Connector.connect(expand0.outputs.output, join.inputs[0])
        Connector.connect(expand1.outputs.output, join.inputs[1])
        Connector.connect(expand2.outputs.output, join.inputs[2])
        Connector.connect(expand3.outputs.output, join.inputs[3])

        Connector.connect(join.outputs.output, expert.inputs.sp.data_input)