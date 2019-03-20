import logging
from typing import List, Optional

from eval_utils import run_just_model
from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.se_node_group import SeNodeGroup
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group import \
    Nc1r1GroupWithAdapter
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.significant_nodes import ConvLayer
from torchsim.topologies.toyarch_groups.ncm_group import NCMGroup

logger = logging.getLogger(__name__)


class Task0TaSeTopology(Topology):
    """ Topology that connects SE-Task0-Dataset -> conv-fully TA
    """

    def __init__(self,
                 top_layer_params: Optional[MultipleLayersParams] = MultipleLayersParams(),
                 conv_layers_params: MultipleLayersParams = MultipleLayersParams(),
                 model_seed: int = 321,

                 # DATASET
                 image_size=SeDatasetSize.SIZE_24,
                 baseline_seed: int = 123,
                 class_filter: List[int] = None,
                 random_order: bool = False,
                 noise_amp: float = 0.0
                 ):
        """
        Constructor of the TA topology which should solve the Task0.

        Args:
            model_seed: seed of the model
            image_size: size of the dataset image
            class_filter: filters the classes in the dataset
            baseline_seed: seed for the baseline nodes
        """

        super().__init__('cuda')

        layer_sizes = conv_layers_params.read_list_of_params('n_cluster_centers')
        if top_layer_params is not None:
            layer_sizes += top_layer_params.read_list_of_params('n_cluster_centers')

        self.se_group = SeNodeGroup(baseline_seed=baseline_seed,
                                    layer_sizes=layer_sizes,
                                    class_filter=class_filter,
                                    image_size=image_size,
                                    random_order=random_order,
                                    noise_amp=noise_amp)

        self.add_node(self.se_group)

        if top_layer_params is None:
            self.model = NCMGroup(conv_layers_params=conv_layers_params,
                                  image_size=(image_size.value, image_size.value, 3),
                                  model_seed=model_seed)
        else:
            self.model = Nc1r1GroupWithAdapter(conv_layers_params=conv_layers_params,
                                               top_layer_params=top_layer_params,
                                               num_labels=20,
                                               image_size=(image_size.value, image_size.value, 3),
                                               model_seed=model_seed)

        self.add_node(self.model)

        Connector.connect(
            self.se_group.outputs.image,
            self.model.inputs.image
        )

        if isinstance(self.model, Nc1r1GroupWithAdapter):
            Connector.connect(
                self.se_group.outputs.labels,
                self.model.inputs.label
            )

    def restart(self):
        pass


if __name__ == '__main__':

    num_conv_layers = 1
    use_top_layer = False

    cp = MultipleLayersParams()
    cp.compute_reconstruction = True
    cp.conv_classes = ConvLayer
    cp.sp_buffer_size = 6000
    cp.sp_batch_size = 4000
    cp.learning_rate = 0.1
    cp.cluster_boost_threshold = 1000
    cp.max_encountered_seqs = 2000
    cp.max_frequent_seqs = 1000
    cp.seq_lookahead = 1
    cp.seq_length = 5

    if num_conv_layers == 2:
        cp.n_cluster_centers = [100, 230]

        cp.rf_size = (8, 8)
        cp.rf_stride = (8, 8)
        cp.num_conv_layers = 2
    else:
        cp.n_cluster_centers = 200
        cp.rf_size = (8, 8)
        cp.rf_stride = (8, 8)

    tp = None
    if use_top_layer:
        tp = MultipleLayersParams()
        tp.n_cluster_centers = 20
        tp.sp_buffer_size = 3000
        tp.sp_batch_size = 2000
        tp.learning_rate = 0.1
        tp.cluster_boost_threshold = 1000

    class_f = [1, 2, 3, 4]

    params = [
        {
            'conv_layers_params': cp,
            'top_layer_params': tp,
            'image_size': SeDatasetSize.SIZE_64,
            'class_filter': class_f,
            'model_seed': None,
            'baseline_seed': None,
            'noise_amp': 0.0,
            'random_order': False
        }
    ]

    run_just_model(Task0TaSeTopology(**params[0]), gui=True, persisting_observer_system=True)
