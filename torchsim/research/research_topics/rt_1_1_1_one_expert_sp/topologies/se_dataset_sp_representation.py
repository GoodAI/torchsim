from torchsim.core.datasets.dataset_se_base import SeDatasetSize, DatasetSeBase
from torchsim.core.nodes.dataset_se_navigation_node import DatasetSeNavigationNode, DatasetSENavigationParams, \
    SamplingMethod
from torchsim.core.nodes.expand_node import ExpandNode
from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.utils.seed_utils import set_global_seeds


class SEDatasetSPRepresentationTopology(Topology):
    """
    A model which receives data from the SE dataset and learns spatial representation from this.
    """

    _node_se_dataset: DatasetSeNavigationNode
    _node_spatial_pooler: SpatialPoolerFlockNode

    sp_params: ExpertParams
    se_world_params: DatasetSENavigationParams

    def __init__(self, seed: int = None):
        super().__init__('cuda')

        self.se_world_params = DatasetSENavigationParams(dataset_size=SeDatasetSize.SIZE_24)
        self.se_world_params.sampling_method = SamplingMethod.ORDERED

        self.sp_params = ExpertParams()
        self.sp_params.n_cluster_centers = 10
        self.sp_params.spatial.input_size = \
            self.se_world_params.dataset_dims[0] * \
            self.se_world_params.dataset_dims[1] * \
            DatasetSeBase.N_CHANNELS
        self.sp_params.flock_size = 3
        self.sp_params.spatial.buffer_size = 100
        self.sp_params.spatial.batch_size = 45
        self.sp_params.spatial.cluster_boost_threshold = 30

        # create the node instances
        se_dataset = DatasetSeNavigationNode(self.se_world_params, seed=seed)
        expand_node = ExpandNode(dim=0,
                                 desired_size=self.sp_params.flock_size)

        sp_node = SpatialPoolerFlockNode(self.sp_params, seed=seed)

        self.add_node(se_dataset)
        self.add_node(expand_node)
        self.add_node(sp_node)

        Connector.connect(se_dataset.outputs.image_output, expand_node.inputs.input)
        Connector.connect(expand_node.outputs.output, sp_node.inputs.sp.data_input)

        set_global_seeds(seed)
