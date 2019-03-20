from math import sqrt

from torchsim.core.datasets.dataset_se_base import SeDatasetSize, DatasetSeBase
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes.dataset_se_navigation_node import DatasetSeNavigationNode, DatasetSENavigationParams, SamplingMethod
from torchsim.core.nodes.random_number_node import RandomNumberNode
from torchsim.core.nodes.receptive_field_node import ReceptiveFieldNode
from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode
from torchsim.research.research_topics.rt_1_1_1_one_expert_sp.topologies.mnist_sp_topology import MnistSpTopology
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.topologies.benchmark_lrf_flock_topology import \
    compute_lrf_params

IMAGE_SIZE = SeDatasetSize.SIZE_24


class SeDatasetSpTopology(Topology):

    _sp_params: ExpertParams
    _dataset_params: DatasetSENavigationParams

    node_dataset: DatasetSeNavigationNode
    node_sp: SpatialPoolerFlockNode

    output_dimension: int  # total output dimension of the flock (and therefore the RandomNumberNode)

    @staticmethod
    def get_se_params(random_order: bool, no_landmarks: int):
        se_params = DatasetSENavigationParams(IMAGE_SIZE)
        if random_order:
            se_params.sampling_method = SamplingMethod.RANDOM_ORDER
        else:
            se_params.sampling_method = SamplingMethod.ORDERED

        # configure the segments
        assert sqrt(float(no_landmarks)).is_integer()
        no_segments = int(sqrt(no_landmarks))
        se_params.horizontal_segments = no_segments
        se_params.vertical_segments = no_segments

        return se_params

    def __init__(self,
                 dataset_seed: int = 123,
                 model_seed: int = 321,
                 baseline_seed: int = 333,
                 num_cc: int = 100,
                 batch_s: int = 300,
                 cbt: int = 1000,
                 lr=0.1,
                 no_landmarks: int = 100,
                 rand_order: bool = False,
                 mbt: int = 1000):
        super().__init__(device="cuda")

        flock_size = 1  # TODO flock_size > 1 not supported by the adapter yet

        flock_input_size = IMAGE_SIZE.value * IMAGE_SIZE.value * DatasetSeBase.N_CHANNELS
        flock_input_size_tuple, flock_output_size = compute_lrf_params(IMAGE_SIZE.value,
                                                                       IMAGE_SIZE.value,
                                                                       DatasetSeBase.N_CHANNELS,
                                                                       eoy=1, eox=1)

        # define params
        self._sp_params = MnistSpTopology.get_sp_params(
            num_cluster_centers=num_cc,
            cluster_boost_threshold=cbt,
            learning_rate=lr,
            buffer_size=2*batch_s,
            batch_size=batch_s,
            input_size=flock_input_size,
            flock_size=flock_size,
            max_boost_time=mbt
        )

        self.output_dimension = flock_size * num_cc

        self._se_params = SeDatasetSpTopology.get_se_params(random_order=rand_order, no_landmarks=no_landmarks)

        # define nodes
        self.node_sp = SpatialPoolerFlockNode(self._sp_params.clone(), seed=model_seed)
        self._lrf_node = ReceptiveFieldNode(flock_input_size_tuple, flock_output_size)
        self.node_dataset = DatasetSeNavigationNode(self._se_params, dataset_seed)
        self.node_random = RandomNumberNode(upper_bound=self.output_dimension, seed=baseline_seed)

        # add nodes and connect the graph
        self.add_node(self.node_dataset)
        self.add_node(self.node_sp)
        self.add_node(self._lrf_node)
        self.add_node(self.node_random)

        # connect SEDataset->LRF->SP
        Connector.connect(
            self.node_dataset.outputs.image_output,
            self._lrf_node.inputs[0])
        Connector.connect(
            self._lrf_node.outputs[0],
            self.node_sp.inputs.sp.data_input)



