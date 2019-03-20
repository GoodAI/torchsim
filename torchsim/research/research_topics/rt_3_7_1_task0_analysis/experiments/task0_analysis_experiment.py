
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Any

from eval_utils import parse_test_args, run_experiment_with_ui, run_experiment
from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.eval2.experiment import Experiment
from torchsim.core.eval2.experiment_controller import TrainTestComponentParams
from torchsim.core.eval2.experiment_runner_params import ExperimentParams
from torchsim.core.eval2.scaffolding import TopologyScaffoldingFactory
from torchsim.core.nodes import DatasetSeObjectsNode
from torchsim.research.experiment_templates2.task0_ta_analysis_template import Task0TaAnalysisTemplate, Task0TaAnalysisParams
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.se_node_group import SeNodeGroup
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.node_groups.ta_multilayer_classification_group import \
    TaMultilayerClassificationGroup
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.topologies.task0_ta_analysis_topology import \
    Task0TaAnalysisTopology
from torchsim.significant_nodes import ConvLayer

logger = logging.getLogger(__name__)


@dataclass
class Params:
    experiment_params: Task0TaAnalysisParams
    train_test_params: TrainTestComponentParams


debug_params = Params(Task0TaAnalysisParams(measurement_period=1,
                                            num_classes=DatasetSeObjectsNode.label_size(),
                                            num_layers=2,  # has to be overwritten by actual num layers later
                                            sp_evaluation_period=2,
                                            show_conv_agreements=False),
                      TrainTestComponentParams(num_testing_phases=3,
                                               num_testing_steps=6,
                                               overall_training_steps=30))

middle_params = Params(Task0TaAnalysisParams(measurement_period=1,
                                             num_classes=DatasetSeObjectsNode.label_size(),
                                             num_layers=2,
                                             sp_evaluation_period=2,
                                             show_conv_agreements=False),
                       TrainTestComponentParams(num_testing_phases=6,
                                                num_testing_steps=200,
                                                overall_training_steps=1500))

full_params = Params(Task0TaAnalysisParams(measurement_period=1,
                                           num_classes=DatasetSeObjectsNode.label_size(),
                                           num_layers=2,
                                           sp_evaluation_period=1000,
                                           show_conv_agreements=False),
                     TrainTestComponentParams(num_testing_phases=15,
                                              num_testing_steps=800,
                                              overall_training_steps=40000))


def run_measurement(name, topology_parameters, args, exp_pars):
    """"Runs the experiment with specified params, see the parse_test_args method for arguments"""

    scaffolding = TopologyScaffoldingFactory(Task0TaAnalysisTopology,
                                             se_group=SeNodeGroup,
                                             model=TaMultilayerClassificationGroup)

    template = Task0TaAnalysisTemplate("Task 0 layer-wise stats and classification accuracy",
                                       exp_pars.experiment_params,
                                       exp_pars.train_test_params)

    runner_parameters = ExperimentParams(max_steps=exp_pars.train_test_params.max_steps,
                                         save_cache=args.save,
                                         load_cache=args.load,
                                         clear_cache=args.clear,
                                         calculate_statistics=not args.computation_only,
                                         experiment_folder=args.alternative_results_folder)

    experiment = Experiment(template, scaffolding, topology_parameters, runner_parameters)

    if args.run_gui:
        run_experiment_with_ui(experiment)
    else:
        logger.info(f'Running model: {name}')
        run_experiment(experiment)

    if args.show_plots:
        plt.show()


def run_opp(args, num_conv_layers: int, exp_params, top_cc: int = 150):
    name = f"OPP-influence_num_cc{top_cc}"

    cp = MultipleLayersParams()
    cp.compute_reconstruction = False
    cp.conv_classes = ConvLayer
    cp.sp_buffer_size = 6000
    cp.sp_batch_size = 1500  # original 4000
    cp.learning_rate = 0.10001
    cp.cluster_boost_threshold = 1000
    cp.max_encountered_seqs = 2000
    cp.max_frequent_seqs = 1000
    cp.seq_lookahead = 1
    cp.seq_length = 4  # note: might use also 5 (as in older experiments)

    if num_conv_layers == 2:
        cp.n_cluster_centers = [100, 230]

        cp.rf_size = [(8, 8), (4, 4)]
        cp.rf_stride = [(8, 8), (1, 1)]

        cp.num_conv_layers = 2
    else:
        cp.n_cluster_centers = 200

        cp.rf_size = (8, 8)
        cp.rf_stride = (8, 8)
        cp.num_conv_layers = 1

    tp = MultipleLayersParams()
    tp.n_cluster_centers = top_cc
    tp.sp_buffer_size = 3000
    tp.sp_batch_size = 1500  #
    tp.learning_rate = 0.15
    tp.cluster_boost_threshold = 1000
    tp.compute_reconstruction = True

    cf_easy = [1, 2, 3, 4]
    size = SeDatasetSize.SIZE_64
    size_int = (size.value, size.value, 3)

    params = [
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {'conv_layers_params': cp.change(opp=0),
                      'top_layer_params': tp,
                      'image_size': size_int}
        },
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {'conv_layers_params': cp.change(opp=0.05),
                      'top_layer_params': tp,
                      'image_size': size_int}
        },
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {'conv_layers_params': cp.change(opp=0.5),
                      'top_layer_params': tp,
                      'image_size': size_int}
        },
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {'conv_layers_params': cp.change(opp=0.95),
                      'top_layer_params': tp,
                      'image_size': size_int}
        },
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {'conv_layers_params': cp.change(opp=1),
                      'top_layer_params': tp,
                      'image_size': size_int}
        },
        # {
        #     'se_group': {'class_filter': cf_easy,
        #                  'image_size': size},
        #     'model': {'conv_layers_params': cp.change(opp=0.3),
        #               'top_layer_params': tp,
        #               'image_size': size_int}
        # },
        # {
        #     'se_group': {'class_filter': cf_easy,
        #                  'image_size': size},
        #     'model': {'conv_layers_params': cp.change(opp=0.7),
        #               'top_layer_params': tp,
        #               'image_size': size_int}
        # }
    ]

    exp_params.experiment_params.num_layers = num_conv_layers + 1  # needs to be there

    run_measurement(name, params, args, exp_params)


def run_num_cc_on_top(args, num_conv_layers: int, exp_params, opp: float = 1.0):
    name = "sp_size_on_top"

    cp = MultipleLayersParams()
    cp.compute_reconstruction = False
    cp.conv_classes = ConvLayer
    cp.sp_buffer_size = 6000
    cp.sp_batch_size = 4000
    cp.learning_rate = 0.1
    cp.cluster_boost_threshold = 1000
    cp.max_encountered_seqs = 2000
    cp.max_frequent_seqs = 1000
    cp.seq_lookahead = 1
    cp.seq_length = 5
    cp.opp = opp

    if num_conv_layers == 2:
        cp.n_cluster_centers = [100, 230]

        cp.rf_size = [(8, 8), (4, 4)]
        cp.rf_stride = [(8, 8), (1, 1)]

        cp.num_conv_layers = 2
    else:
        cp.n_cluster_centers = 200

        cp.rf_size = (8, 8)
        cp.rf_stride = (8, 8)
        cp.num_conv_layers = 1

    tp = MultipleLayersParams()
    tp.n_cluster_centers = 200
    tp.sp_buffer_size = 3000
    tp.sp_batch_size = 2000
    tp.learning_rate = 0.15
    tp.cluster_boost_threshold = 1000
    tp.compute_reconstruction = True

    cf_easy = [1, 2, 3, 4]
    size = SeDatasetSize.SIZE_64
    size_int = (size.value, size.value, 3)

    params = [
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {'conv_layers_params': cp,
                      'top_layer_params': tp.change(n_cluster_centers=200),
                      'image_size': size_int}
        },
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {'conv_layers_params': cp,
                      'top_layer_params': tp.change(n_cluster_centers=500),
                      'image_size': size_int}
        },
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {'conv_layers_params': cp,
                      'top_layer_params': tp.change(n_cluster_centers=1000),
                      'image_size': size_int}
        },
    ]

    exp_params.experiment_params.num_layers = num_conv_layers + 1  # needs to be there

    run_measurement(name, params, args, exp_params)


def run_good_topology(args, exp_params):
    name = "good topology"

    params = good_one_layer_config_for_four_objects()

    exp_params.experiment_params.num_layers = 2

    run_measurement(name, params, args, exp_params)


def good_one_layer_config_for_four_objects() -> List[Dict[str, Any]]:
    """A topology which might achieve 100% SE accuracy on 4 objects"""

    cp = MultipleLayersParams()
    cp.compute_reconstruction = False
    cp.conv_classes = ConvLayer
    cp.sp_buffer_size = 6000
    cp.sp_batch_size = 4000
    cp.learning_rate = 0.1
    cp.cluster_boost_threshold = 1000
    cp.max_encountered_seqs = 2000
    cp.max_frequent_seqs = 1000
    cp.seq_lookahead = 1
    cp.seq_length = 5

    cp.n_cluster_centers = 200

    cp.rf_size = (8, 8)
    cp.rf_stride = (8, 8)
    cp.num_conv_layers = 1

    tp = MultipleLayersParams()
    tp.n_cluster_centers = 150
    tp.sp_buffer_size = 3000
    tp.sp_batch_size = 2000
    tp.learning_rate = 0.15
    tp.cluster_boost_threshold = 1000
    tp.compute_reconstruction = True

    cf_easy = [1, 2, 3, 4]
    size = SeDatasetSize.SIZE_64
    size_int = (size.value, size.value, 3)

    params = [
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {'conv_layers_params': cp.change(opp=0.5),
                      'top_layer_params': tp,
                      'image_size': size_int}
        }
    ]

    return params


def run_rf_size(args, exp_params, opp: float = 1.0):
    name = "rf_size"

    cp = MultipleLayersParams()
    cp.compute_reconstruction = False
    cp.conv_classes = ConvLayer
    cp.sp_buffer_size = 6000
    cp.sp_batch_size = 4000
    cp.learning_rate = 0.1
    cp.cluster_boost_threshold = 1000
    cp.max_encountered_seqs = 2000
    cp.max_frequent_seqs = 1000
    cp.seq_lookahead = 1
    cp.seq_length = 5
    cp.opp = opp

    cp.n_cluster_centers = 200

    cp.rf_size = (8, 8)
    cp.rf_stride = (8, 8)
    cp.num_conv_layers = 1

    tp = MultipleLayersParams()
    tp.n_cluster_centers = 150
    tp.sp_buffer_size = 3000
    tp.sp_batch_size = 2000
    tp.learning_rate = 0.15
    tp.cluster_boost_threshold = 1000
    tp.compute_reconstruction = True

    cf_easy = [1, 2, 3, 4]
    size = SeDatasetSize.SIZE_64
    size_int = (size.value, size.value, 3)

    params = [
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {'conv_layers_params': cp.change(rf_stride=([(8, 8)])),
                      'top_layer_params': tp,
                      'image_size': size_int}
        },
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {'conv_layers_params': cp.change(rf_stride=([(4, 4)])),
                      'top_layer_params': tp,
                      'image_size': size_int}
        },
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {'conv_layers_params': cp.change(rf_stride=([(2, 2)])),
                      'top_layer_params': tp,
                      'image_size': size_int}
        },
    ]

    exp_params.experiment_params.num_layers = 2  # needs to be there

    run_measurement(name, params, args, exp_params)


def run_debug_base(args, num_conv_layers: int, exp_params):
    name = "Learning-rate-debug_ncl_"+str(num_conv_layers)

    cp = MultipleLayersParams()
    cp.compute_reconstruction = False
    cp.conv_classes = ConvLayer
    cp.sp_buffer_size = 6000
    cp.sp_batch_size = 4000
    cp.learning_rate = 0.10005
    cp.cluster_boost_threshold = 1000
    cp.max_encountered_seqs = 2000
    cp.max_frequent_seqs = 1000
    cp.seq_lookahead = 1
    cp.seq_length = 5

    if num_conv_layers == 2:
        cp.n_cluster_centers = [100, 230]

        cp.rf_size = [(8, 8), (4, 4)]
        cp.rf_stride = [(8, 8), (1, 1)]

        cp.num_conv_layers = 2
    else:
        cp.n_cluster_centers = 200

        cp.rf_size = (8, 8)
        cp.rf_stride = (8, 8)
        cp.num_conv_layers = 1

    tp = MultipleLayersParams()
    tp.n_cluster_centers = 100
    tp.sp_buffer_size = 3000
    tp.sp_batch_size = 2000
    tp.learning_rate = 0.01
    tp.cluster_boost_threshold = 1000
    tp.compute_reconstruction = True

    cf_easy = [1, 2, 3, 4]
    size = SeDatasetSize.SIZE_64
    size_int = (size.value, size.value, 3)

    params = [
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {'conv_layers_params': cp,
                      'top_layer_params': tp,
                      'image_size': size_int}
        }
    ]

    exp_params.experiment_params.num_layers = num_conv_layers + 1  # needs to be there

    run_measurement(name, params, args, exp_params)


if __name__ == '__main__':
    arg = parse_test_args()

    pars = debug_params

    # run_good_topology(arg, pars)

    # run_opp(arg, 1, pars, top_cc=50)
    # run_opp(arg, 1, pars, top_cc=100)
    run_opp(arg, 1, pars, top_cc=150)
    run_opp(arg, 1, pars, top_cc=300)
    run_opp(arg, 1, pars, top_cc=450)
    run_opp(arg, 1, pars, top_cc=600)
    # run_rf_size(arg, pars, opp=0.5)
    # run_rf_size(arg, 1, pars, 0.95)
    # run_opp(arg, 2, pars)
    # run_num_cc_on_top(arg, 1, pars, opp=0.95)

    # run_num_cc_on_top(arg, 1, pars)
    # run_num_cc_on_top(arg, 2, pars)