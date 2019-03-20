from eval_utils import run_just_model, parse_test_args
from torchsim.research.experiment_templates.task0_train_test_classification_acc_template import \
    Task0TrainTestClassificationAccTemplate

from torchsim.research.research_topics.rt_1_1_4_task0_experiments.experiments.experiment_template_params import \
    TrainTestExperimentTemplateParams
from torchsim.research.research_topics.rt_2_1_2_learning_rate.adapters.modular.classification_accuracy_modular_adapter import \
    ClassificationAccuracyModularAdapter
from torchsim.research.research_topics.rt_2_1_2_learning_rate.topologies.task0_ta_se_topology import Task0TaSeTopology

debug_params = TrainTestExperimentTemplateParams(
    measurement_period=1,
    sp_evaluation_period=2,

    overall_training_steps=300,
    num_testing_steps=200,
    num_testing_phases=10)


full_params = TrainTestExperimentTemplateParams(
    measurement_period=1,
    sp_evaluation_period=100,  # original value: 200

    overall_training_steps=25000,  # original value: 100'000
    num_testing_steps=500,  # original value: 1000
    num_testing_phases=25)  # original value: 20


def run_measurement_with_params(name,
                                params,
                                args,
                                exp_pars: TrainTestExperimentTemplateParams = None,
                                topology_class=Task0TaSeTopology):
    """Run a given experiment with given commandline params, topology and experiment params"""

    if exp_pars is None:
        exp_pars = full_params

    experiment = Task0TrainTestClassificationAccTemplate(
        ClassificationAccuracyModularAdapter(),
        topology_class,  # Task0TaSeTopology or Task0NnTopology supported for now
        params,
        overall_training_steps=exp_pars.overall_training_steps,
        num_testing_steps=exp_pars.num_testing_steps,
        num_testing_phases=exp_pars.num_testing_phases,
        num_classes=20,  # this should match the configuration of SE or SE dataset that we'll use
        num_layers=2,
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
        disable_plt_show=True
    )

    if args.run_gui:
        run_just_model(topology_class(**params[0]), gui=True)
    else:
        print(f'======================================= Measuring model: {name}')
        experiment.run()


def run_measurement(name, params, args, debug: bool = False):
    """"Runs the experiment with specified params, see the parse_test_args method for arguments"""

    exp_pars = debug_params if debug else None
    run_measurement_with_params(name, params, args, exp_pars=exp_pars)


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


def run_debug(args, debug: bool = True):
    name = "Learning-rate-debug"
    params = [
        {'lr': [0.79, 0.79], 'batch_s': [300, 150], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.7, 0.7], 'batch_s': [30, 15], 'num_cc': [300, 200], 'class_filter': [1, 2, 3, 4]},
    ]
    run_measurement(name, params, args, debug)


if __name__ == '__main__':
    arg = parse_test_args()

    run_debug(arg, debug=True)
    # run_learning_rate(arg, False)
    # run_cc(arg, debug=False)
