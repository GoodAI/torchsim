from eval_utils import run_just_model, parse_test_args
from torchsim.core.nodes.dataset_se_objects_node import DatasetSeObjectsNode

from torchsim.research.experiment_templates.task0_online_learning_template import Task0OnlineLearningTemplate

# ideally at least 30000 steps here according to results
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.adapters.task0_conv_wide_adapter import \
    Task0ConvWideAdapter
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.conv_wide_two_layer_topology import \
    ConvWideTwoLayerTopology

DEF_MAX_STEPS = 70000


def run_measurement(name, params, args, max_steps: int = DEF_MAX_STEPS):
    """Runs the experiment with specified params, see the parse_test_args method for arguments."""

    experiment = Task0OnlineLearningTemplate(
        Task0ConvWideAdapter(),
        ConvWideTwoLayerTopology,
        params,
        max_steps=max_steps,
        num_training_steps=max_steps // 10 * 8,  # (1 tenth of time is used for testing)
        num_classes=DatasetSeObjectsNode.label_size(),  # TODO make this somehow better
        num_layers=2,               # TODO parametrized
        measurement_period=5,       # 4
        sliding_window_size=300,    # 300
        sliding_window_stride=100,   # 50
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
        run_just_model(ConvWideTwoLayerTopology(**params[0]), gui=True)
    else:
        print(f'======================================= Measuring model: {name}')
        experiment.run()


def run_learning_rate(args, max_steps):
    name = "learning_rate"

    params = [
        {'lr': [0.2, 0.05], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.2, 0.1], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.2, 0.2], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.2, 0.3], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.2, 0.4], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.2, 0.5], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.2, 0.6], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.2, 0.7], 'class_filter': [1, 2, 3, 4]},
    ]
    run_measurement(name, params, args, max_steps)


def run_learning_rate_slower(args, max_steps):
    name = "lr"

    params = [
        {'lr': [0.005, 0.0001], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.01, 0.0005], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.02, 0.001], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.02, 0.005], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.025, 0.01], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.025, 0.02], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.025, 0.05], 'class_filter': [1, 2, 3, 4]},
        {'lr': [0.2, 0.05], 'class_filter': [1, 2, 3, 4]},
    ]
    run_measurement(name, params, args, max_steps)


if __name__ == '__main__':
    arg = parse_test_args()

    run_learning_rate_slower(arg, DEF_MAX_STEPS)

