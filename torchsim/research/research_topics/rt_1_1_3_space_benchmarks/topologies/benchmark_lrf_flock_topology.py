from torchsim.core.datasets.dataset_se_base import SeDatasetSize, DatasetSeBase
from torchsim.core.graph import Topology
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes.dataset_se_navigation_node import DatasetSENavigationParams, SamplingMethod


def compute_flock_sizes(sy: int, sx: int, num_channels: int, eoy: int, eox: int) -> [int, int]:
    input_size = (sx // eox) * (sy // eoy) * num_channels
    flock_size = eox * eoy
    return flock_size, input_size


def setup_flock_params(no_clusters, buffer_size, batch_size, tp_learn_period, max_enc_seq, flock_size, input_size):
    params = ExpertParams()
    params.n_cluster_centers = no_clusters
    params.flock_size = flock_size

    params.spatial.input_size = input_size
    params.spatial.buffer_size = buffer_size
    params.spatial.batch_size = batch_size
    params.spatial.cluster_boost_threshold = 30

    # tp params used only if run_just_sp=False
    params.temporal.seq_length = 3
    params.temporal.seq_lookahead = 1
    params.temporal.buffer_size = 160
    params.temporal.batch_size = 150
    params.temporal.learning_period = tp_learn_period + params.temporal.seq_lookbehind
    params.temporal.incoming_context_size = 10
    params.temporal.max_encountered_seqs = max_enc_seq
    params.temporal.n_frequent_seqs = max_enc_seq // 2
    params.temporal.forgetting_limit = 2000
    return params


def compute_lrf_params(sy: int, sx: int, num_channels: int, eoy: int, eox: int):
    flock_input_size = (sy, sx, num_channels)
    flock_output_size = (sy//eoy, sx//eox)
    return flock_input_size, flock_output_size


def init_se_dataset_world_params(random_order: bool):
    """Returns world_params, sy, sx, no_channels."""
    se_world_params = DatasetSENavigationParams(dataset_size=SeDatasetSize.SIZE_64)
    if random_order:
        se_world_params.sampling_method = SamplingMethod.RANDOM_ORDER
    return se_world_params, se_world_params.dataset_dims[0], se_world_params.dataset_dims[1], DatasetSeBase.N_CHANNELS


class BenchmarkLrfFlockTopology(Topology):
    _eox: int
    _eoy: int

    def __init__(self, eox: int = 1, eoy: int = 1, device='cuda'):
        super().__init__(device)
        self._eox = eox
        self._eoy = eoy
