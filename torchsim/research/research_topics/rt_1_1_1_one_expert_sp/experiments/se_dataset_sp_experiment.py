from eval_utils import run_just_model, parse_test_args

from torchsim.research.experiment_templates.sp_learning_convergence_template import SpLearningConvergenceExperimentTemplate
from torchsim.research.research_topics.rt_1_1_1_one_expert_sp.adapters.se_dataset_sp_learning_convergence_adapter import \
    SeDatasetSpLearningConvergenceTopologyAdapter
from torchsim.research.research_topics.rt_1_1_1_one_expert_sp.topologies.se_dataset_sp_topology import SeDatasetSpTopology

DEF_MAX_STEPS = 15000


def read_max_steps(max_steps: int = None):
    if max_steps is None:
        return DEF_MAX_STEPS
    return max_steps


def run_measurement(name, params, args, max_steps: int = None):
    """Runs the experiment with specified params, see the parse_test_args method for arguments."""

    experiment = SpLearningConvergenceExperimentTemplate(
        SeDatasetSpLearningConvergenceTopologyAdapter(),
        SeDatasetSpTopology,
        params,
        max_steps=read_max_steps(max_steps),
        num_classes=100,
        measurement_period=1,
        sliding_window_size=200,  # be aware that the mutual information depends on this value
        sliding_window_stride=50,
        sp_evaluation_period=70,
        save_cache=args.save,
        load_cache=args.load,
        clear_cache=args.clear,
        experiment_name=name,
        computation_only=args.computation_only,
        experiment_folder=args.alternative_results_folder,
        disable_plt_show=True
    )

    if args.run_gui:
        run_just_model(SeDatasetSpTopology(**params[0]), gui=True)
    else:
        print(f'======================================= Measuring model: {name}')
        experiment.run()


def run_num_cc(args, max_steps):
    name = "num_cc"
    params = [
        {'num_cc': 500},
        {'num_cc': 400},
        {'num_cc': 300},
        {'num_cc': 200},
        {'num_cc': 100},
        {'num_cc': 10},
        {'num_cc': 5},
        {'num_cc': 2},
        {'num_cc': 1}
    ]
    run_measurement(name, params, args, max_steps)


def run_learning_rate(args, max_steps):
    """Learning rate vs mutual info."""
    name = "learning_rate"

    params = [
        {'lr': 0.01},
        {'lr': 0.1},
        {'lr': 0.2},
        {'lr': 0.5},
        {'lr': 0.7},
        {'lr': 1}
    ]

    run_measurement(name, params, args, max_steps)


def run_learning_rate_rand(args, max_steps):
    """Learning rate vs mutual info."""
    name = "lr_rand"

    params = [
        {'lr': 0.01, 'rand_order': True},
        {'lr': 0.1, 'rand_order': True},
        {'lr': 0.2, 'rand_order': True},
        {'lr': 0.5, 'rand_order': True},
        {'lr': 0.7, 'rand_order': True},
        {'lr': 1, 'rand_order': True}
    ]

    run_measurement(name, params, args, max_steps)


def run_different_seeds(args, max_steps):
    """The same parameters, just different seeds"""
    name = "init_robust"

    params = [
        {'model_seed': None, 'dataset_seed': None},
        {'model_seed': None, 'dataset_seed': None},
        {'model_seed': None, 'dataset_seed': None},
        {'model_seed': None, 'dataset_seed': None},
        {'model_seed': None, 'dataset_seed': None},
        {'model_seed': 1, 'dataset_seed': 3},
        {'model_seed': 121, 'dataset_seed': 3497},
    ]

    run_measurement(name, params, args, max_steps)


def run_cluster_boost_threshold(args, max_steps):
    """Different values of the cluster boost threshold parameter"""

    name = "cluster_b_thr"

    params = [
        {'cbt': 1},
        {'cbt': 10},
        {'cbt': 50},
        {'cbt': 100},
        {'cbt': 150},
        {'cbt': 500},
        {'cbt': 700},
        {'cbt': 1500},
        {'cbt': 3000},
    ]
    run_measurement(name, params, args, max_steps)


def run_batch_size(args, max_steps):
    """Batch_size vs mutual info"""

    # batch too small (with the right learning rate should cause low stability and low MI
    # maybe just for non-iid data (SEDataset(s))
    name = "batch_size"

    params = [
        {'batch_s': 5},
        {'batch_s': 50},
        {'batch_s': 100},
        {'batch_s': 500},
        {'batch_s': 1000},
    ]
    run_measurement(name, params, args, max_steps)


def run_batch_size_longer(args, max_steps):
    """Batch_size vs mutual info"""

    name = "batch_longer"

    params = [
        {'batch_s': 5},
        {'batch_s': 500},
        {'batch_s': 2000},
        {'batch_s': 3000},
        {'batch_s': 5000},
        {'batch_s': 9000}
    ]
    run_measurement(name, params, args, max_steps)


def run_debug(args, max_steps: int):
    name = "debug-exp"
    params = [
        {'lr': 0.05},
        {'lr': 0.1},
    ]
    run_measurement(name, params, args, max_steps)


if __name__ == '__main__':
    arg = parse_test_args()

    num_steps = 15000

    run_different_seeds(arg, num_steps)
    run_num_cc(arg, num_steps)
    run_cluster_boost_threshold(arg, num_steps)
    run_batch_size(arg, num_steps)
    run_batch_size_longer(arg, num_steps)
    run_learning_rate(arg, num_steps)
    run_learning_rate_rand(arg, num_steps)
