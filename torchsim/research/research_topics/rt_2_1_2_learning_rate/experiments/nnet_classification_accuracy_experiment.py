from eval_utils import parse_test_args
from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.experiments.experiment_template_params import \
    TrainTestExperimentTemplateParams

from torchsim.research.research_topics.rt_2_1_2_learning_rate.experiments.ta_classification_accuracy_experiment import \
    run_measurement_with_params
from torchsim.research.research_topics.rt_2_1_2_learning_rate.topologies.task0_nn_topology import Task0NnTopology


debug_params = TrainTestExperimentTemplateParams(
    measurement_period=1,
    sp_evaluation_period=1,

    overall_training_steps=5000,
    num_testing_steps=3000,
    num_testing_phases=4)

my_full_params = TrainTestExperimentTemplateParams(
    measurement_period=1,
    sp_evaluation_period=100,

    overall_training_steps=500000,  # original value: 100'000
    num_testing_steps=4000,
    num_testing_phases=20)


def run_measurement(name, params, args, debug: bool=False):
    """"Runs the experiment with specified params, see the parse_test_args method for arguments"""

    if debug:
        exp_pars = debug_params
    else:
        exp_pars = my_full_params

    run_measurement_with_params(name,
                                params,
                                args,
                                exp_pars=exp_pars,
                                topology_class=Task0NnTopology)


def run_learning_rate(args, debug: bool=False):
    """Learning rate vs accuracy"""
    name = "learning_rate"

    params = [
        {'lr': 0.0005},
        {'lr': 0.001},
        {'lr': 0.011},
        {'lr': 0.11},
        {'lr': 0.21},
    ]

    run_measurement(name, params, args, debug)


# def run_default_params(args, debug: bool=False):
#     name = "nnet-default"
#     params = [{}]
#     run_measurement(name, params, args, debug, use_custom_full_params=True)


def run_model_seed(args, debug: bool=False):
    name = "seed"

    params = [
        {'model_seed': None},
        {'model_seed': 1},
        # {'model_seed': 2},
        # {'model_seed': 3},
        # {'model_seed': 44},
    ]
    run_measurement(name, params, args, debug)


def run_batch_size(args, debug: bool = False):
    name = "batch_size"

    params = [
        {'buffer_s': 512},
        {'buffer_s': 1024},
        {'buffer_s': 2048},
    ]
    run_measurement(name, params, args, debug)


def run_class_filter(args, debug: bool = True):
    name = "random_order_test"
    params = [
        {'class_filter': [2, 3, 7], 'random_order': False},
        {'class_filter': [2, 3, 7], 'random_order': True},
        {'random_order': False, 'lr': 0.0011},
        {'random_order': True, 'lr': 0.0011}
    ]
    run_measurement(name, params, args, debug)


def run_full(args, debug: bool = True):
    name = "full_experiment"
    params = [
        {'random_order': False},
        {'random_order': True}
    ]
    run_measurement(name, params, args, debug)


def run_seeds(args, debug: bool = True):
    name = "seeds"
    params = [
        {'random_order': False},
        {'random_order': True},
        {'random_order': False, 'model_seed': 123},
        {'random_order': True, 'model_seed': 123},
        {'random_order': False, 'model_seed': 7},
        {'random_order': True, 'model_seed': 7}
    ]
    run_measurement(name, params, args, debug)


def run_seeds_high_res(args, debug: bool = True):
    name = "seeds"
    params = [
        {'image_size': SeDatasetSize.SIZE_64, 'random_order': False},
        {'image_size': SeDatasetSize.SIZE_64, 'random_order': True},
        {'image_size': SeDatasetSize.SIZE_64, 'random_order': False, 'model_seed': 123},
        {'image_size': SeDatasetSize.SIZE_64, 'random_order': True, 'model_seed': 123},
        {'image_size': SeDatasetSize.SIZE_64, 'random_order': False, 'model_seed': 7},
        {'image_size': SeDatasetSize.SIZE_64, 'random_order': True, 'model_seed': 7}
    ]
    run_measurement(name, params, args, debug)


def run_grayscale(args, debug: bool = True):
    name = "grayscale"
    params = [
        {'image_size': SeDatasetSize.SIZE_64, 'random_order': False, 'use_grayscale': True, 'use_se': True},
    ]
    run_measurement(name, params, args, debug)


def run_debug(args, debug: bool = True):
    name = "debug"
    params = [
        {'image_size': SeDatasetSize.SIZE_24,'lr': 0.011, 'class_filter': [2, 3, 7], 'num_epochs': 10, 'random_order': True}
    ]
    run_measurement(name, params, args, debug)


if __name__ == '__main__':
    """This thing measures the classification accuracies of the NNet baseline on the Task0"""

    arg = parse_test_args()

    # run_class_filter(arg, debug=True)

    # run_class_filter(arg, debug=True)
    # run_full(arg, debug=False)
    # run_seeds(arg, debug=True)
    # run_debug(arg, debug=True)
    # run_seeds_high_res(arg, debug=False)
    run_grayscale(arg, debug=False)
    # run_default_params(arg, debug=True)
    # run_debug(arg, debug=True)
    # run_model_seed(arg, debug=True)
    # run_default_params(arg)
    # run_batch_size(arg)
    # run_learning_rate(arg)
