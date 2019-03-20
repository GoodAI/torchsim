from eval_utils import run_just_model, parse_test_args
from torchsim.core.nodes.dataset_se_objects_node import DatasetSeObjectsNode

from torchsim.research.experiment_templates.task0_online_learning_template import Task0OnlineLearningTemplate

# ideally at least 30000 steps here according to results
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.adapters.task0_narrow_adapter import Task0NarrowAdapter
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.task0_narrow_topology import Task0NarrowTopology

DEF_MAX_STEPS = 50000


def read_max_steps(max_steps: int = None):
    if max_steps is None:
        return DEF_MAX_STEPS
    return max_steps


def run_measurement(name, params, args, max_steps: int = None):
    """Runs the experiment with specified params, see the parse_test_args method for arguments."""

    experiment = Task0OnlineLearningTemplate(
        Task0NarrowAdapter(),
        Task0NarrowTopology,
        params,
        max_steps=read_max_steps(max_steps),
        num_training_steps=read_max_steps(max_steps) // 10 * 8,  # (1 tenth of time is used for testing)
        num_classes=DatasetSeObjectsNode.label_size(),  # TODO make this somehow better
        num_layers=3,  # TODO parametrized
        measurement_period=4,
        sliding_window_size=300,
        sliding_window_stride=50,
        sp_evaluation_period=200,
        save_cache=args.save,
        load_cache=args.load,
        clear_cache=args.clear,
        experiment_name=name,
        computation_only=args.computation_only,
        experiment_folder=args.alternative_results_folder,
        disable_plt_show=True,
        just_hide_labels=True   #
    )

    if args.run_gui:
        run_just_model(Task0NarrowTopology(**params[0]), gui=True)
    else:
        print(f'======================================= Measuring model: {name}')
        experiment.run()


def run_num_cc(args, max_steps):
    name = "num_cc"
    params = [
        {'num_cc': [400, 150, 150]},
        {'num_cc': [400, 200, 200]},
        {'num_cc': [400, 200, 150]},
        {'num_cc': [400, 150, 50]},
    ]
    run_measurement(name, params, args, max_steps)


def run_num_cc_short(args, max_steps):
    name = "num_cc"
    params = [
        {'num_cc': [400, 150, 150], 'lr': [0.5, 0.5, 0.5], 'batch_s': [1000, 1000, 700]},
        # {'num_cc': [400, 200, 100], 'lr': [0.5, 0.5, 0.5], 'batch_s': [1000, 1000, 700]},
    ]
    run_measurement(name, params, args, max_steps)


def run_learning_rate(args, max_steps):
    """Learning rate vs mutual info."""
    name = "learning_rate"

    params = [
        {'lr': [0.05, 0.05, 0.05]},
        {'lr': [0.02, 0.02, 0.02]},
        # {'lr': [0.5, 0.5, 0.5]},
        # {'lr': [0.3, 0.3, 0.3]},
    ]

    run_measurement(name, params, args, max_steps)


def run_label_scale(args, max_steps: int):
    name = "label_scale"
    params = [
        {'label_scale': 0.15},
        {'label_scale': 0.25},
        {'label_scale': 0.5},
        {'label_scale': 1},
        {'label_scale': 2},
        {'label_scale': 50},
        {'label_scale': 100},
        # {'label_scale': 1000}
    ]
    run_measurement(name, params, args, max_steps)


def run_seq_len(args, max_steps: int):
    name = "seq_len"
    params = [
        {'seq_len': 3},
        {'seq_len': 4},
        {'seq_len': 5}
    ]
    run_measurement(name, params, args, max_steps)


def run_batch_size(args, max_steps):
    """Batch_size vs mutual info."""

    # batch too small (with the right learning rate should cause low stability and low MI
    # maybe just for non-iid data (SEDataset(s))
    name = "batch_size"

    params = [
        {'batch_s': [2000, 1500, 1000]},
        {'batch_s': [1500, 1000, 500]},
        {'batch_s': [1500, 700, 300]},
    ]
    run_measurement(name, params, args, max_steps)


def run_debug(args, max_steps: int):
    name = "lr"
    params = [
        {'lr': [0.99, 0.99, 0.99], 'batch_s': [300, 300, 300]},
        {'lr': [0.2, 0.2, 0.2], 'batch_s': [2000, 1500, 1000], 'num_cc': [300, 200, 100]},
        # {'lr': 0.2}
    ]
    run_measurement(name, params, args, max_steps)


if __name__ == '__main__':
    arg = parse_test_args()

    # num_steps = 100000
    # num_steps =3000
    # num_steps = 1000
    # num_steps = 800
    # num_steps = 500
    num_steps = None

    # run_debug(arg, num_steps)
    #run_num_cc_short(arg, num_steps)

    #run_label_scale(arg, num_steps)
    #run_seq_len(arg, num_steps)
    #run_batch_size(arg, num_steps)
    #run_num_cc(arg, num_steps)
    run_learning_rate(arg, num_steps)

    # run_different_seeds(arg, num_steps)
    # run_cluster_boost_threshold(arg, num_steps)
