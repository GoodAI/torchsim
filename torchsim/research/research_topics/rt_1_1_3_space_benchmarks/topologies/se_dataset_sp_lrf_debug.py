from eval_utils import parse_test_args
from torchsim.core.nodes.dataset_se_navigation_node import DatasetSeNavigationNode
from torchsim.core.nodes.receptive_field_node import ReceptiveFieldNode
from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode
from torchsim.core.graph.connection import Connector
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.experiments.se_dataset_sp_running_stats_experiment import \
    run_measurement
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.nodes.rgb_debug_node import RgbDebugNode
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.topologies.benchmark_lrf_flock_topology import \
    BenchmarkLrfFlockTopology, compute_flock_sizes, setup_flock_params, compute_lrf_params, init_se_dataset_world_params
from torchsim.utils.seed_utils import set_global_seeds


class SeDatasetSpLrfDebug(BenchmarkLrfFlockTopology):
    """
    A model which receives data from the SE dataset and learns spatial representation from this.
    """

    def __init__(self,
                 seed: int=0,
                 device: str = 'cuda',
                 eox: int = 2,
                 eoy: int = 2,
                 num_cc=30,
                 batch_s=150):
        super().__init__(eox, eoy)

        # compute/setup parameters of the model
        se_world_params, self._sy, self._sx, self._no_channels = init_se_dataset_world_params(random_order=False)
        flock_size, input_size = compute_flock_sizes(self._sy, self._sx, self._no_channels, self._eoy, self._eox)
        expert_params = setup_flock_params(no_clusters=num_cc,
                                           buffer_size=batch_s * 2,
                                           batch_size=batch_s,
                                           tp_learn_period=100,
                                           max_enc_seq=1000,
                                           flock_size=flock_size,
                                           input_size=input_size)
        flock_input_size, flock_output_size = compute_lrf_params(self._sy, self._sx, self._no_channels, self._eoy,
                                                                 self._eox)
        # crate nodes
        self._se_dataset = DatasetSeNavigationNode(se_world_params, seed=seed)
        self._lrf_node = ReceptiveFieldNode(flock_input_size, flock_output_size)
        self._sp_node = SpatialPoolerFlockNode(expert_params, seed=seed)
        self._rgb_debug_node = RgbDebugNode(input_dims=flock_input_size, channel_first=False)  # just a debug

        self.add_node(self._se_dataset)
        self.add_node(self._lrf_node)
        self.add_node(self._sp_node)
        self.add_node(self._rgb_debug_node)

        # connect Dataset -> LRF -> SP
        Connector.connect(
            self._se_dataset.outputs.image_output,
            self._lrf_node.inputs[0])
        Connector.connect(
            self._lrf_node.outputs[0],
            self._sp_node.inputs.sp.data_input)

        # Dataset -> debug
        Connector.connect(
            self._se_dataset.outputs.image_output,
            self._rgb_debug_node.inputs.input)

        set_global_seeds(seed)


def run_debug(args, max_steps):
    name = 'num_cluster_centers'
    params = [
        {'eox': 2, 'eoy': 2, 'num_cc': 10},
        {'eox': 2, 'eoy': 2, 'num_cc': 25}]
    run_measurement(name, params, args, max_steps)


if __name__ == '__main__':
    """
    Just for the debugging purposes
    """
    arg = parse_test_args()
    ms = 200

    run_debug(arg, ms)


