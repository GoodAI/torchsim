import logging

from eval_utils import run_just_model
from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.eval2.scaffolding import TopologyScaffoldingFactory
from torchsim.core.eval2.train_test_switchable import TrainTestSwitchable
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.se_node_group import SeNodeGroup
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.node_groups.multilayer_model_group import MultilayerModelGroup
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.node_groups.ta_multilayer_classification_group import TaMultilayerClassificationGroup
from torchsim.significant_nodes import ConvLayer

logger = logging.getLogger(__name__)


class Task0TaAnalysisTopology(Topology, TrainTestSwitchable):
    """ Topology that connects SE-Task0-Dataset -> conv-fully TA
    """

    def __init__(self, se_group: SeNodeGroup, model: MultilayerModelGroup):

        super().__init__('cuda')
        self.se_group = se_group
        self.model = model

        self.add_node(self.se_group)
        self.add_node(self.model)

        Connector.connect(
            self.se_group.outputs.image,
            self.model.inputs.image
        )

        Connector.connect(
            self.se_group.outputs.labels,
            self.model.inputs.label
        )

    # train/test
    def switch_to_training(self):
        self.se_group.switch_dataset_training(True)
        self.model.model_switch_to_training()

    def switch_to_testing(self):
        self.se_group.switch_dataset_training(False)
        self.model.model_switch_to_testing()

    def is_learning(self) -> bool:
        return self.model.is_learning()


if __name__ == '__main__':

    num_conv_layers = 1
    use_top_layer = True

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

    if not use_top_layer:
        tp = None
    else:
        tp = MultipleLayersParams()
        tp.n_cluster_centers = 20
        tp.sp_buffer_size = 3000
        tp.sp_batch_size = 2000
        tp.learning_rate = 0.1
        tp.cluster_boost_threshold = 1000
        tp.compute_reconstruction = True

    cf_easy = [1, 2, 3, 4]

    params = [
        {'se_group': {'class_filter': cf_easy},
         'model': {'conv_layers_params': cp,
                   'top_layer_params': tp}},
        {'se_group': {'class_filter': cf_easy},
         'model': {'conv_layers_params': cp.change(learning_rate=0.7, sp_batch_size=30, n_cluster_centers=300),
                   'top_layer_params': tp.change(learning_rate=0.7, sp_batch_size=15, n_cluster_centers=200)}}
    ]

    # TODO merging common params not supported yeat
    common_params = [
        {
            'conv_layers_params': cp,
            'top_layer_params': tp,
            'image_size': SeDatasetSize.SIZE_64,
            'class_filter': cf_easy,
            'model_seed': None,
            'baseline_seed': None,
            'noise_amp': 0.0,
            'random_order': False
        }
    ]

    scaffolding = TopologyScaffoldingFactory(Task0TaAnalysisTopology,
                                             se_group=SeNodeGroup,
                                             model=TaMultilayerClassificationGroup)

    run_just_model(scaffolding.create_topology(**params[0]), gui=True, persisting_observer_system=True)
