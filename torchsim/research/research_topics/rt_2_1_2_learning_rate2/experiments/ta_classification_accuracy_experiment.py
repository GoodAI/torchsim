import logging
from dataclasses import dataclass

from eval_utils import parse_test_args, run_experiment_with_ui
from torchsim.core.eval2.experiment_controller import TrainTestComponentParams
from torchsim.core.eval2.experiment import Experiment
from torchsim.core.eval2.experiment_runner_params import ExperimentParams
from torchsim.core.eval2.scaffolding import TopologyScaffoldingFactory
from torchsim.core.nodes import DatasetSeObjectsNode
from torchsim.research.experiment_templates2.task0_train_test_classification_acc_template import \
    Task0TrainTestClassificationAccTemplate, Task0TrainTestClassificationAccParams
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.se_node_group import SeNodeGroup
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.research.research_topics.rt_2_1_2_learning_rate2.topologies.classification_accuracy_modular_topology import \
    ClassificationAccuracyModularTopology
from torchsim.research.research_topics.rt_2_1_2_learning_rate2.node_groups.ta_multilayer_node_group import \
    Nc1r1ClassificationGroup

logger = logging.getLogger(__name__)


@dataclass
class Params:
    experiment_params: Task0TrainTestClassificationAccParams
    train_test_params: TrainTestComponentParams


debug_params = Params(Task0TrainTestClassificationAccParams(measurement_period=1,
                                                            num_classes=DatasetSeObjectsNode.label_size(),
                                                            sp_evaluation_period=2),
                      TrainTestComponentParams(num_testing_phases=6,
                                               num_testing_steps=12,
                                               overall_training_steps=360))

full_params = Params(Task0TrainTestClassificationAccParams(measurement_period=1,
                                                           num_classes=DatasetSeObjectsNode.label_size(),
                                                           sp_evaluation_period=100),
                     TrainTestComponentParams(num_testing_phases=25,
                                              num_testing_steps=500,
                                              overall_training_steps=25000))


def run_measurement(name, topology_parameters, args, debug: bool = False):
    """"Runs the experiment with specified params, see the parse_test_args method for arguments"""
    exp_pars = debug_params if debug else full_params

    scaffolding = TopologyScaffoldingFactory(ClassificationAccuracyModularTopology,
                                             se_group=SeNodeGroup,
                                             model=Nc1r1ClassificationGroup)

    template = Task0TrainTestClassificationAccTemplate("Task 0 Train/Test Classification Accuracy",
                                                       exp_pars.experiment_params,
                                                       exp_pars.train_test_params)

    runner_parameters = ExperimentParams(max_steps=exp_pars.train_test_params.max_steps,
                                         save_cache=args.save,
                                         load_cache=args.load,
                                         clear_cache=args.clear,
                                         calculate_statistics=not args.computation_only,
                                         experiment_folder=args.alternative_results_folder)

    experiment = Experiment(template, scaffolding, topology_parameters, runner_parameters)

    run_experiment_with_ui(experiment)


def run_learning_rate(args, debug: bool = False):
    """Learning rate vs mutual info"""
    name = "learning_rate"

    params = [
        {'lr': [0.5, 0.5]},
        {'lr': [0.1, 0.1]},
    ]

    run_measurement(name, params, args, debug)


def run_cc(args, debug: bool = False):
    name = "num_cc"
    params = [
        {'lr': [0.02, 0.05], 'num_cc': [150, 40], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        # {'lr': [0.05, 0.02], 'num_cc': [150, 100], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        {'lr': [0.05, 0.02], 'num_cc': [250, 150], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        # {'lr': [0.05, 0.02], 'num_cc': [250, 200], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        {'lr': [0.05, 0.02], 'num_cc': [350, 200], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        # {'lr': [0.05, 0.02], 'num_cc': [350, 250], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        {'lr': [0.05, 0.02], 'num_cc': [350, 300], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        # {'lr': [0.05, 0.02], 'num_cc': [500, 300], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        # {'lr': [0.05, 0.02], 'num_cc': [500, 400], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        # {'lr': [0.05, 0.02], 'num_cc': [500, 600], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        # {'lr': [0.05, 0.02], 'num_cc': [500, 800], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
        # {'lr': [0.05, 0.02], 'num_cc': [500, 1000], 'class_filter': [1, 2, 3, 4, 5, 6, 7]},
    ]

    run_measurement(name, params, args, debug)


def run_debug_comparison(args, debug: bool = True):
    name = "Learning-rate-debug"
    # params = [
    #     {'se_group': {'class_filter': [1, 2, 3, 4]},
    #      'model': {'lr': [0.79, 0.79], 'batch_s': [300, 150]}},
    #     {'se_group': {'class_filter': [1, 2, 3, 4]},
    #      'model': {'lr': [0.7, 0.7], 'batch_s': [30, 15], 'num_cc': [300, 200]}}
    # ]
    params = [
        {'se_group': {'class_filter': [1, 2, 3, 4]},
         'model': {
             'conv_layers_params': MultipleLayersParams(learning_rate=0.79, sp_batch_size=300),
             'top_layer_params': MultipleLayersParams(learning_rate=0.79, sp_batch_size=150)}},
        {'se_group': {'class_filter': [1, 2, 3, 4]},
         'model': {
             'conv_layers_params': MultipleLayersParams(learning_rate=0.7, sp_batch_size=30, n_cluster_centers=300),
             'top_layer_params': MultipleLayersParams(learning_rate=0.7, sp_batch_size=15, n_cluster_centers=200)}}
    ]
    run_measurement(name, params, args, debug)


if __name__ == '__main__':
    arg = parse_test_args()

    run_debug_comparison(arg, debug=True)
    # run_learning_rate(arg, False)
    # run_cc(arg, debug=False)
