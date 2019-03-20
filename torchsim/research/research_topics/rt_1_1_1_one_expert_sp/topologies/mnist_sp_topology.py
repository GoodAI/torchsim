from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import RandomNumberNode, DatasetMNISTParams, DatasetMNISTNode, ReceptiveFieldNode, \
    SpatialPoolerFlockNode
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.topologies.benchmark_lrf_flock_topology import \
    compute_lrf_params

NUM_CLASSES = 10


class MnistSpTopology(Topology):

    _sp_params: ExpertParams
    _mnist_params: DatasetMNISTParams

    node_mnist: DatasetMNISTNode
    node_sp: SpatialPoolerFlockNode

    output_dimension: int  # total output dimension of the flock (and therefore the RandomNumberNode)

    @staticmethod
    def get_sp_params(num_cluster_centers: int,
                      cluster_boost_threshold: int,
                      learning_rate: float,
                      buffer_size: int,
                      batch_size: int,
                      input_size: int,
                      flock_size: int,
                      max_boost_time):
        params = ExpertParams()
        params.n_cluster_centers = num_cluster_centers
        params.spatial.input_size = input_size
        params.flock_size = flock_size

        params.spatial.buffer_size = buffer_size
        params.spatial.batch_size = batch_size
        params.spatial.cluster_boost_threshold = cluster_boost_threshold
        params.spatial.max_boost_time = max_boost_time  # should be bigger than any cluster_boost_threshold
        params.spatial.learning_rate = learning_rate

        return params

    @staticmethod
    def get_mnist_params(examples_per_cl: int = None):
        params = DatasetMNISTParams()
        params.examples_per_class = examples_per_cl
        params.one_hot_labels = False
        return params

    def __init__(self,
                 dataset_seed: int = 123,
                 model_seed: int = 321,
                 baseline_seed: int = 333,
                 num_cc: int = 10,
                 batch_s: int = 300,
                 cbt: int = 1000,
                 lr=0.1,
                 examples_per_cl: int=None,
                 mbt: int = 1000):
        super().__init__("cuda")

        flock_size = 1  # TODO flock_size > 1 not supported by the adapter yet

        # define params
        self._sp_params = MnistSpTopology.get_sp_params(
            num_cluster_centers=num_cc,
            cluster_boost_threshold=cbt,
            learning_rate=lr,
            buffer_size=2*batch_s,
            batch_size=batch_s,
            input_size=28*28,
            flock_size=flock_size,
            max_boost_time=mbt
        )

        self.output_dimension = flock_size * num_cc

        _mnist_params = MnistSpTopology.get_mnist_params(examples_per_cl)
        flock_input_size, flock_output_size = compute_lrf_params(28, 28, 1, eoy=1, eox=1)

        # define nodes
        self.node_sp = SpatialPoolerFlockNode(self._sp_params.clone(), seed=model_seed)
        self._lrf_node = ReceptiveFieldNode(flock_input_size, flock_output_size)
        self.node_mnist = DatasetMNISTNode(params=_mnist_params, seed=dataset_seed)
        self.node_random = RandomNumberNode(upper_bound=self.output_dimension, seed=baseline_seed)

        # add nodes and connect the graph
        self.add_node(self.node_mnist)
        self.add_node(self.node_sp)
        self.add_node(self._lrf_node)
        self.add_node(self.node_random)

        # connect MNIST->LRF->SP
        Connector.connect(
            self.node_mnist.outputs.data,
            self._lrf_node.inputs[0])
        Connector.connect(
            self._lrf_node.outputs[0],
            self.node_sp.inputs.sp.data_input)

