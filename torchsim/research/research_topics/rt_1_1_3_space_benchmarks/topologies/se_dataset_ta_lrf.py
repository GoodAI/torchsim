from torchsim.core.nodes.dataset_se_navigation_node import DatasetSeNavigationNode
from torchsim.core.nodes.expert_node import ExpertFlockNode
from torchsim.core.nodes.receptive_field_node import ReceptiveFieldNode
from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import NUMBER_OF_CONTEXT_TYPES
from torchsim.core.nodes.constant_node import ConstantNode
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.topologies.benchmark_lrf_flock_topology import \
    BenchmarkLrfFlockTopology, init_se_dataset_world_params, compute_flock_sizes, setup_flock_params, compute_lrf_params
from torchsim.utils.seed_utils import set_global_seeds


class SeDatasetTaLrf(BenchmarkLrfFlockTopology):
    """
    A model which receives data from the SE dataset and learns spatial and temporal patterns
    """

    def __init__(self,
                 run_just_sp: bool = False,
                 seed: int = None,
                 device: str = 'cuda',
                 eox: int = 2,
                 eoy: int = 2,
                 num_cc: int = 100,
                 batch_s=300,
                 tp_learn_period=50,
                 tp_max_enc_seq=1000):
        super().__init__(eox, eoy)

        # compute/setup parameters of the model
        se_world_params, self._sy, self._sx, self._no_channels = init_se_dataset_world_params(random_order=False)
        flock_size, input_size = compute_flock_sizes(self._sy, self._sx, self._no_channels, self._eoy, self._eox)
        expert_params = setup_flock_params(no_clusters=num_cc,
                                           buffer_size=batch_s * 2,
                                           batch_size=batch_s,
                                           tp_learn_period=tp_learn_period,
                                           max_enc_seq=tp_max_enc_seq,
                                           flock_size=flock_size,
                                           input_size=input_size)
        flock_input_size, flock_output_size = compute_lrf_params(self._sy, self._sx, self._no_channels, self._eoy,
                                                                 self._eox)

        # crate nodes
        self._se_dataset = DatasetSeNavigationNode(se_world_params, seed=seed)
        self._lrf_node = ReceptiveFieldNode(flock_input_size, flock_output_size)
        if run_just_sp:
            self._flock_node = SpatialPoolerFlockNode(expert_params, seed=seed)
        else:
            self._flock_node = ExpertFlockNode(expert_params, seed=seed)

        self._zero_context = ConstantNode(shape=(expert_params.flock_size, expert_params.temporal.n_providers, NUMBER_OF_CONTEXT_TYPES,
                                                 expert_params.temporal.incoming_context_size), constant=0)

        # add nodes to the graph
        self.add_node(self._se_dataset)
        self.add_node(self._lrf_node)
        self.add_node(self._flock_node)
        self.add_node(self._zero_context)

        # connect Dataset -> LRF -> SP
        Connector.connect(
            self._se_dataset.outputs.image_output,
            self._lrf_node.inputs[0])
        Connector.connect(
            self._lrf_node.outputs[0],
            self._flock_node.inputs.sp.data_input)

        if not run_just_sp:
            Connector.connect(
                self._zero_context.outputs.output,
                self._flock_node.inputs.tp.context_input)

        # prepare for run
        set_global_seeds(seed)
        self._last_step_duration = 0
