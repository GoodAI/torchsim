from eval_utils import run_just_model, parse_test_args
from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.eval.experiment_template_base import ExperimentTemplateBase
from torchsim.core.nodes.dataset_se_objects_node import DatasetSeObjectsNode

from torchsim.research.experiment_templates.task0_train_test_learning_rate_template import Task0TrainTestLearningRateTemplate
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.experiments.experiment_template_params import \
    TrainTestExperimentTemplateParams
from torchsim.research.research_topics.rt_2_1_2_learning_rate.adapters.modular.learning_rate_ta_modular_adapter import \
    LearningRateTaModularAdapter
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.research.research_topics.rt_2_1_2_learning_rate.topologies.task0_ta_se_topology import Task0TaSeTopology
from torchsim.significant_nodes import ConvLayer, SpConvLayer

debug_params = TrainTestExperimentTemplateParams(
    measurement_period=1,
    sp_evaluation_period=10,

    overall_training_steps=300,
    num_testing_steps=20,
    num_testing_phases=3)

# mid experiment params
# debug_params = TrainTestExperimentTemplateParams(
#     measurement_period=1,
#     sp_evaluation_period=20,
#
#     overall_training_steps=20000,
#     num_testing_steps=400,
#     num_testing_phases=20)

# full experiment params
full_params = TrainTestExperimentTemplateParams(
    measurement_period=1,
    sp_evaluation_period=250,

    overall_training_steps=50000,  # original value: 100'000
    num_testing_steps=1000,
    num_testing_phases=20)


def run_measurement(name, params, args, debug: bool = False, num_layers: int = 3):
    """"Runs the experiment with specified params, see the parse_test_args method for arguments"""

    exp_pars = debug_params if debug else full_params

    experiment = Task0TrainTestLearningRateTemplate(
        LearningRateTaModularAdapter(),
        Task0TaSeTopology,
        params,
        overall_training_steps=exp_pars.overall_training_steps,
        num_testing_steps=exp_pars.num_testing_steps,
        num_testing_phases=exp_pars.num_testing_phases,
        num_classes=DatasetSeObjectsNode.label_size(),
        num_layers=num_layers,
        measurement_period=exp_pars.measurement_period,
        sliding_window_size=1,  # not used
        sliding_window_stride=1,  # not used
        sp_evaluation_period=exp_pars.sp_evaluation_period,
        save_cache=args.save,
        load_cache=args.load,
        clear_cache=args.clear,
        experiment_name=name,
        computation_only=args.computation_only,
        experiment_folder=args.alternative_results_folder,
        disable_plt_show=True,
        show_conv_agreements=False
    )

    if args.run_gui:
        run_just_model(Task0TaSeTopology(**params[0]), gui=True)
    else:
        print(f'======================================= Measuring model: {name}')
        experiment.run()


def run_cluster_boost_threshold(args, debug: bool=False):
    """Different values for the cluster boost threshold"""

    name = "cluster_boost_thr"

    # TODO test some boost_thresholds with default parameters found in the learning_rate
    params = [
        #{'cbt': (1000, 1), 'num_cc': [200, 500], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        #{'cbt': (1000, 200), 'num_cc': [200, 500], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        {'cbt': (1000, 100000), 'num_cc': [200, 500], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
    ]
    run_measurement(name, params, args, debug)


def run_learning_rate_determine_max_steps(args, debug: bool = False):
    """Just run this for many steps, estimate where it converges and then run others with optimal no. steps"""
    name = "learning_rate"
    params = [
        # good performance tested
        # {'lr': [0.2, 0.05], 'num_cc': [200, 500], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        # slowest rates used
        {'lr': [0.1, 0.005], 'num_cc': [200, 500], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
    ]

    run_measurement(name, params, args, debug)


def run_learning_rate(args, debug: bool=False):
    """Learning rate vs mutual info"""
    name = "learning_rate"

    params = [
        # good performance tested
        {'lr': [0.2, 0.05], 'num_cc': [200, 500], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},

        # different LR in the conv layer
        {'lr': [0.005, 0.05], 'num_cc': [200, 500], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        {'lr': [0.05, 0.05], 'num_cc': [200, 500], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        {'lr': [0.4, 0.05], 'num_cc': [200, 500], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},

        # different LRs in the top expert
        {'lr': [0.1, 0.005], 'num_cc': [200, 500], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        {'lr': [0.1, 0.2], 'num_cc': [200, 500], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        #{'lr': [0.1, 0.01], 'num_cc': [200, 500], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
    ]

    run_measurement(name, params, args, debug)


def run_more_classes(args, debug: bool = False):
    name = "more_classes"

    use = True

    params = [
        {'lr': [0.1, 0.1], 'num_cc': [100, 150], 'class_filter': [2, 3, 7]},
        {'lr': [0.1, 0.1], 'num_cc': [100, 250], 'class_filter': [2, 3, 7]},
        {'lr': [0.1, 0.1], 'num_cc': [100, 300], 'class_filter': [2, 3, 7]},
        {'lr': [0.1, 0.1], 'num_cc': [100, 300], 'class_filter': [2, 3, 7, 19]},
        {'lr': [0.1, 0.1], 'num_cc': [100, 400], 'class_filter': [2, 3, 7, 19]},
        {'lr': [0.1, 0.1], 'num_cc': [100, 400], 'class_filter': [2, 3, 7, 19, 0]},
        {'lr': [0.1, 0.1], 'num_cc': [100, 450], 'class_filter': [2, 3, 7, 19, 0]},
    ]
    run_measurement(name, params, args, debug)


def run_more_classes_tp(args, debug: bool = False):
    name = "more_classes_tp"

    use = False

    params = [
        {'lr': [0.1, 0.1], 'num_cc': [100, 150], 'conv_classes': [SpConvLayer], 'class_filter': [2, 3, 7]},
        {'lr': [0.1, 0.1], 'num_cc': [100, 250], 'conv_classes': [SpConvLayer], 'class_filter': [2, 3, 7]},
        {'lr': [0.1, 0.1], 'num_cc': [100, 300], 'conv_classes': [SpConvLayer], 'class_filter': [2, 3, 7]},
        {'lr': [0.1, 0.1], 'num_cc': [100, 300], 'conv_classes': [SpConvLayer], 'class_filter': [2, 3, 7, 19]},
        {'lr': [0.1, 0.1], 'num_cc': [100, 400], 'conv_classes': [SpConvLayer], 'class_filter': [2, 3, 7, 19]},
        {'lr': [0.1, 0.1], 'num_cc': [100, 400], 'conv_classes': [SpConvLayer], 'class_filter': [2, 3, 7, 19, 0]},
        {'lr': [0.1, 0.1], 'num_cc': [100, 450], 'conv_classes': [SpConvLayer], 'class_filter': [2, 3, 7, 19, 0]},
        {'lr': [0.1, 0.1], 'num_cc': [100, 450], 'conv_classes': [SpConvLayer], 'class_filter': [2, 3, 7, 19, 0, 10]},
        {'lr': [0.1, 0.1], 'num_cc': [100, 500], 'conv_classes': [SpConvLayer], 'class_filter': [2, 3, 7, 19, 0, 10]},
    ]
    run_measurement(name, params, args, debug)


def run_more_classes_three_layers(args, just_sp: bool = True, debug: bool = False):
    name = "more_classes_three"

    cf = 'class_filter'

    three = [2, 3, 7]
    four = [2, 3, 7, 19]
    five = [2, 3, 7, 19, 0]
    six = [2, 3, 7, 19, 0, 10]

    params = [
        {'num_cc': [150, 150, 150], cf: three},
        {'num_cc': [150, 150, 500], cf: three},

        {'num_cc': [150, 150, 500], cf: four},
        {'num_cc': [150, 150, 700], cf: four},

        {'num_cc': [150, 150, 700], cf: five},
        {'num_cc': [150, 150, 900], cf: five},

        {'num_cc': [150, 150, 700], cf: six},
        {'num_cc': [150, 150, 900], cf: six},
    ]

    common_params = {
        'lr': [0.1, 0.1, 0.1],
        'batch_s': [3000, 3000, 1000],
        'buffer_s': [6000, 6000, 6000],
        'conv_classes': [ConvLayer, ConvLayer],
        'experts_on_x': [4, 4],
        'cbt': [1000, 1000, 1000]
    }

    for param in params:
        for key, value in common_params.items():
            param[key] = value

    run_measurement(name, params, args, debug)


def run_eox_three_layers_full(args, use_sp: bool = True, debug: bool = False):
    name = "experts_on_x_full"

    cf = 'class_filter'
    us = 'use_sp'

    params = [
        #{'experts_on_x': [4, 4], cf: None, us: True},
        #{'experts_on_x': [8, 8], cf: None, us: True},
        #{'experts_on_x': [12, 12], cf: three, 'num_cc': [50, 50, 500]}, # unspecified launch failure at 37522
        #{'experts_on_x': [4, 4], cf: None, us: False},
        {'experts_on_x': [8, 8], cf: None, us: [use_sp, use_sp]},
    ]

    common_params = {
        'lr': [0.1, 0.1, 0.1],
        'batch_s': [3000, 3000, 1000],
        'buffer_s': [6000, 6000, 6000],
        #'experts_on_x': [4, 4],
        'cbt': [1000, 1000, 1000],
        'num_cc': [70, 70, 500]
    }

    for param in params:
        for key, value in common_params.items():
            param[key] = value

    run_measurement(name, params, args, debug)


def run_eox_three_layers(args, debug: bool = False):
    name = "experts_on_x"

    cf = 'class_filter'

    three = [2, 3, 7]
    four = [2, 3, 7, 19]
    five = [2, 3, 7, 19, 0]
    six = [2, 3, 7, 19, 0, 10]

    params = [
        {'experts_on_x': [4, 4], cf: three},
        {'experts_on_x': [8, 8], cf: three},
        #{'experts_on_x': [12, 12], cf: three, 'num_cc': [50, 50, 500]}, # unspecified launch failure at 37522

        {'experts_on_x': [4, 4], cf: four},
        {'experts_on_x': [8, 8], cf: four},
        #{'experts_on_x': [12, 12], cf: four, 'num_cc': [50, 50, 500]},

        {'experts_on_x': [4, 4], cf: five},
        {'experts_on_x': [8, 8], cf: five},
        #{'experts_on_x': [12, 12], cf: five, 'num_cc': [50, 50, 500]},
    ]

    use = True

    common_params = {
        'lr': [0.1, 0.1, 0.1],
        'batch_s': [3000, 3000, 1000],
        'buffer_s': [6000, 6000, 6000],
        'use_sp': [use, use],
        #'experts_on_x', [4, 4],
        'cbt': [1000, 1000, 1000],
        'num_cc': [100, 100, 500]
    }

    for param in params:
        for key, value in common_params.items():
            param[key] = value

    run_measurement(name, params, args, debug)


def run_debug_II(args, debug: bool = True):
    name = "slower_LR_smaller_eox"

    cf = 'class_filter'

    two = [2, 3]
    three = [2, 3, 7]
    four = [2, 3, 7, 19]
    five = [2, 3, 7, 19, 0]
    six = [2, 3, 7, 19, 0, 10]

    params = [

        # {'experts_on_x': [8, 4], cf: two, 'num_cc': [300, 150, 500], 'conv_classes': [ConvLayer, ConvLayer]},

        {'experts_on_x': [8, 4], cf: two, 'conv_classes': [ConvLayer, SpConvLayer]},
        {'experts_on_x': [8, 4], cf: four, 'conv_classes': [SpConvLayer, ConvLayer]},
        # {'experts_on_x': [8, 4], cf: two, 'conv_classes': [SpConvLayer, ConvLayer]},
        # {'experts_on_x': [8, 4], cf: four, 'conv_classes': [SpConvLayer, ConvLayer]},
    ]

    common_params = {
        'lr': [0.002, 0.002, 0.05],
        'batch_s': [300, 300, 1000],
        'buffer_s': [4000, 4000, 4000],
        # 'conv_classes': [ConvLayer, ConvLayer],
        'cbt': [1000, 1000, 1000],
        'num_cc': [300, 300, 500],
        'image_size': SeDatasetSize.SIZE_64
    }

    for param in params:
        for key, value in common_params.items():
            param[key] = value

    run_measurement(name, params, args, debug)


def run_debug(args, debug: bool = True):
    name = "slower_LR"

    cf = 'class_filter'

    two = [2, 3]
    three = [2, 3, 7]
    four = [2, 3, 7, 19]
    five = [2, 3, 7, 19, 0]
    six = [2, 3, 7, 19, 0, 10]

    params = [
        # {'experts_on_x': [8, 8], cf: two, 'conv_classes': [ConvLayer, ConvLayer]},
        {'experts_on_x': [8, 8], cf: four, 'conv_classes': [ConvLayer, ConvLayer]},

        {'experts_on_x': [8, 8], cf: two, 'conv_classes': [SpConvLayer, ConvLayer]},
        # {'experts_on_x': [8, 8], cf: four, 'conv_classes': [SpConvLayer, ConvLayer]},

    ]

    common_params = {
        'lr': [0.002, 0.002, 0.05],
        'batch_s': [300, 300, 1000],
        'buffer_s': [4000, 4000, 4000],
        'conv_classes': [ConvLayer, ConvLayer],
        'cbt': [1000, 1000, 1000],
        'num_cc': [150, 150, 500],
        'image_size': SeDatasetSize.SIZE_64
    }

    for param in params:
        for key, value in common_params.items():
            param[key] = value

    run_measurement(name, params, args, debug)


def run_simple(args, debug: bool = False):
    name = "simple"
    params = [
        {'lr': [0.2, 0.2], 'num_cc': [100, 250], 'class_filter': [2, 3, 7]},
        {'lr': [0.2, 0.2], 'num_cc': [100, 250], 'class_filter': [2, 3, 7], 'conv_classes': [ConvLayer]},
        # {'lr': [0.1, 0.1, 0.1],
        #  'num_cc': [100, 100, 250],
        #  'batch_s': [3000, 3000, 3000],
        #  'buffer_s': [6000, 6000, 6000],
        #  'cbt': (1000, 1000, 1000),
        #  'use_sp': (False, False),
        #  'experts_on_x': (8, 4),
        #  'class_filter': [2, 3, 7],
        #  },
    ]
    run_measurement(name, params, args, debug)


def run_two_layer_net(args, debug: bool = False):
    name = "two_layer"

    cf = 'class_filter'

    two = [2, 15]
    four = [2, 15, 4, 7]
    six = [2, 15, 4, 7, 10, 19]

    params = [
        {'experts_on_x': [8], cf: six, 'seq_len': 4, 'model_seed': None},
    ]

    common_params = {
        'lr': [0.02, 0.2],
        'batch_s': [2000, 500],
        'buffer_s': [6000, 4000],
        'conv_classes': [ConvLayer],
        'cbt': [1000, 1000],
        'num_cc': [400, 250],
        'image_size': SeDatasetSize.SIZE_64,
        'noise_amp': 0
    }

    for param in params:
        for key, value in common_params.items():
            param[key] = value

    run_measurement(name, params, args, debug=debug, num_layers=2)


def multiple_runs_class_filter_example(args, debug: bool = False):
    name = "example_multiple_runs"

    cp = MultipleLayersParams()
    cp.compute_reconstruction = True
    cp.conv_classes = [ConvLayer, ConvLayer]
    cp.num_conv_layers = 2
    cp.rf_stride = (8, 8)
    cp.rf_size = (8, 8)

    tp = MultipleLayersParams()
    tp.n_cluster_centers = 250

    # class filter changed
    params = [
        {'class_filter': [1, 2]},
        {'class_filter': [1, 2, 3]},
        {'class_filter': [1]}
    ]

    common_params = {
        'conv_layers_params': cp,
        'top_layer_params': tp,
        'image_size': SeDatasetSize.SIZE_64,
        'noise_amp': 0.0,
        'model_seed': None,
        'baseline_seed': None
    }

    # merge the params and common params
    p = ExperimentTemplateBase.add_common_params(params, common_params)

    run_measurement(name, p, args, debug=debug)


def multiple_runs_lr_example(args, debug: bool = False):

    name = "example_multiple_runs"

    default_conv = MultipleLayersParams()
    default_conv.compute_reconstruction = True
    default_conv.conv_classes = [ConvLayer, SpConvLayer]
    default_conv.num_conv_layers = 2

    default_conv.rf_stride = (8, 8)
    default_conv.rf_size = (8, 8)

    tp = MultipleLayersParams()
    tp.n_cluster_centers = 250

    params = [
        {'conv_layers_params': default_conv.change(learning_rate=0.1)},
        {'conv_layers_params': default_conv.change(learning_rate=0.2), 'top_layer_params': tp.change(learning_rate=0.11)},
        {'conv_layers_params': default_conv.change(learning_rate=0.3)}
    ]

    common_params = {
        'top_layer_params': tp,
        'image_size': SeDatasetSize.SIZE_64,
        'noise_amp': 0.0,
        'model_seed': None,
        'baseline_seed': None
    }

    # merge the params and common params
    p = ExperimentTemplateBase.add_common_params(params, common_params)

    run_measurement(name, p, args, debug=debug)


def run_two_layer_net_new(args, debug: bool = False):
    """An example of the new experiment configuration"""

    name = "two_layer_new"

    two = [2, 15]
    four = [2, 15, 4, 7]
    six = [2, 15, 4, 7, 10, 19]

    cp = MultipleLayersParams()
    cp.compute_reconstruction = True
    cp.conv_classes = ConvLayer
    cp.sp_buffer_size = 6000
    cp.sp_batch_size = 4000
    cp.learning_rate = 0.02
    cp.cluster_boost_threshold = 1000
    cp.max_encountered_seqs = 10000
    cp.max_frequent_seqs = 1000
    cp.seq_length = 4
    cp.seq_lookahead = 1
    cp.num_conv_layers = 1

    cp.n_cluster_centers = 400
    cp.rf_size = (8, 8)
    cp.rf_stride = (8, 8)

    tp = MultipleLayersParams()
    tp.n_cluster_centers = 250
    tp.sp_buffer_size = 4000
    tp.sp_batch_size = 500
    tp.learning_rate = 0.2
    tp.cluster_boost_threshold = 1000
    tp.compute_reconstruction = True

    params = [
        {
            'conv_layers_params': cp,
            'top_layer_params': tp,
            'image_size': SeDatasetSize.SIZE_64,
            'class_filter': six,
            'noise_amp': 0.0,
            'model_seed': None,
            'baseline_seed': None
        }
    ]

    run_measurement(name, params, args, debug=debug, num_layers=2)


if __name__ == '__main__':
    arg = parse_test_args()
    #run_more_classes(arg, False)
    #run_more_classes_tp(arg, False)

    # run_debug(arg, debug=True)
    # run_debug_II(arg, debug=True)
    # run_two_layer_net(arg, debug=False)

    # run_two_layer_net_new(arg, debug=True)

    multiple_runs_lr_example(arg, debug=True)
    #run_more_classes_three_layers(arg, debug=False)

    #run_eox_three_layers(arg, debug=False)
    #run_eox_three_layers(arg, debug=True)
    #run_more_classes_three_layers(arg, just_sp=False, debug=False)
    # run_eox_three_layers_full(arg, use_sp=True, debug=False)

    # run_more_classes(arg, debug=True)
    # run_cluster_boost_threshold(arg, False)

    # run_learning_rate_determine_max_steps(arg)
    # run_learning_rate(arg, False)

