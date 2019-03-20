from eval_utils import run_just_model, parse_test_args
from torchsim.core.nodes.dataset_se_objects_node import DatasetSeObjectsNode

from torchsim.research.experiment_templates.task0_online_learning_template import Task0OnlineLearningTemplate

# ideally at least 30000 steps here according to results
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.adapters.task0_conv_wide_adapter import \
    Task0ConvWideAdapter
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.task0_conv_wide_topology import \
    Task0ConvWideTopology
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.task0_narrow_topology import Task0NarrowTopology

DEF_MAX_STEPS = 70000


def read_max_steps(max_steps: int = None):
    if max_steps is None:
        return DEF_MAX_STEPS
    return max_steps


def run_measurement(name, params, args, max_steps: int = None):
    """Runs the experiment with specified params, see the parse_test_args method for arguments"""

    experiment = Task0OnlineLearningTemplate(
        Task0ConvWideAdapter(),
        Task0ConvWideTopology,
        params,
        max_steps=read_max_steps(max_steps),
        num_training_steps=read_max_steps(max_steps) // 10 * 8,  # (1 tenth of time is used for testing)
        num_classes=DatasetSeObjectsNode.label_size(),  # TODO make this somehow better
        num_layers=3,  # TODO parametrized
        measurement_period=4,       # 4
        sliding_window_size=300,    # be aware that the mutual information depends on this value
        sliding_window_stride=50,   # 50
        sp_evaluation_period=200,
        save_cache=args.save,
        load_cache=args.load,
        clear_cache=args.clear,
        experiment_name=name,
        computation_only=args.computation_only,
        experiment_folder=args.alternative_results_folder,
        disable_plt_show=True,
        just_hide_labels=True      # do not switch to testing data, just hide labels?#
    )

    if args.run_gui:
        run_just_model(Task0NarrowTopology(**params[0]), gui=True)
    else:
        print(f'======================================= Measuring model: {name}')
        experiment.run()


def run_num_cc_short(args, max_steps):
    name = "num_cc"
    params = [
        {'num_cc': [70, 150, 150], 'lr': [0.3, 0.3, 0.3], 'batch_s': [2000, 1000, 700]},
        {'num_cc': [150, 150, 150], 'lr': [0.3, 0.3, 0.3], 'batch_s': [2000, 1000, 700]},
        {'num_cc': [150, 200, 200], 'lr': [0.3, 0.3, 0.3], 'batch_s': [2000, 1000, 700]},
        {'num_cc': [150, 300, 300], 'lr': [0.3, 0.3, 0.3], 'batch_s': [2000, 1000, 700]},
        # {'num_cc': [400, 200, 100], 'lr': [0.5, 0.5, 0.5], 'batch_s': [1000, 1000, 700]},
    ]
    run_measurement(name, params, args, max_steps)


def run_num_cc(args, max_steps):
    name = "num_cc"
    params = [
        {'num_cc': [100, 150, 150]},  # 'batch_s': [300, 300, 300]},
        # {'num_cc': [20, 150, 150]},
        # {'num_cc': [30, 150, 150]},
        # {'num_cc': [50, 150, 150]},
        # {'num_cc': [400, 200, 100], 'lr': [0.5, 0.5, 0.5], 'batch_s': [1000, 1000, 700]},
    ]
    run_measurement(name, params, args, max_steps)


def run_learning_rate(args, max_steps):
    """Learning rate vs mutual info."""
    name = "learning_rate"

    params = [
        {'lr': [0.005, 0.005, 0.005]},
        {'lr': [0.01, 0.01, 0.01]},
        {'lr': [0.03, 0.03, 0.03]},
        # {'lr': [0.5, 0.5, 0.5]},
        # {'lr': [0.3, 0.3, 0.3]},
    ]
    run_measurement(name, params, args, max_steps)


def run_multiplier(args, max_steps: int):
    name = "label_scale"
    params = [
        {'label_scale': 1},
        {'label_scale': 50},
        {'label_scale': 500},
        {'label_scale': 1000}
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


def run_debug(args, max_steps: int):
    name = "learning_rate"
    params = [
        {'lr': [0.9, 0.9, 0.9], 'batch_s': [30, 30, 30]},
        # {'lr': [0.9, 0.9, 0.9], 'label_scale': 1000}
    ]
    run_measurement(name, params, args, max_steps)


def run_gui(args):
    name = 'gui'
    params = [
        {'num_cc': [400, 150, 150],
         'lr': [0.5, 0.1, 0.1],
         'batch_s': [3000, 1000, 700],
         'label_scale': 0.1},
    ]


if __name__ == '__main__':
    arg = parse_test_args()

    # num_steps = 100000
    # num_steps = 800
    # num_steps = 20000
    # num_steps = 7000
    num_steps = None

    # run_debug(arg, num_steps)
    run_learning_rate(arg, num_steps)
    #run_multiplier(arg, num_steps)
    #run_num_cc(arg, num_steps)
    # run_num_cc_short(arg, num_steps)
    #run_seq_len(arg, num_steps)

