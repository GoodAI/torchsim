import sys

from eval_utils import parse_test_args, run_just_model
from torchsim.research.se_tasks.experiments.task_1_experiment_template import Task1ExperimentTemplate
from torchsim.research.se_tasks.adapters.task_1_stats_basic_adapter import Task1StatsBasicAdapter
from torchsim.research.se_tasks.topologies.se_task1_basic_topology import SeT1Bt

DEF_MAX_STEPS = sys.maxsize


def read_max_steps(max_steps: int = None):
    if max_steps is None:
        return DEF_MAX_STEPS
    return max_steps


def run_measurement_task1(name, params, args, topology_class, max_steps: int = None):
    """Runs the experiment with specified params, see the parse_test_args method for arguments."""

    experiment = Task1ExperimentTemplate(
        Task1StatsBasicAdapter(),
        topology_class,
        params,
        max_steps=read_max_steps(max_steps),
        measurement_period=1,
        smoothing_window_size=29,
        save_cache=args.save,
        load_cache=args.load,
        clear_cache=args.clear,
        experiment_name=name,
        computation_only=args.computation_only,
        experiment_folder=args.alternative_results_folder
    )

    if args.run_gui:
        run_just_model(SeT1Bt(**params[0]), gui=True)
    else:
        print(f'======================================= Measuring model: {name}')
        experiment.run()


def run_measurements_for_task1(custom_durations: bool = False):
    args = parse_test_args()

    if custom_durations:
        curriculum_tuple = (2001, -1)  # dev curriculum, 15 minutes of runtime (or your own duration)
    else:
        curriculum_tuple = (10010, 1, -1)  # evaluation curriculum with pre-training, 6 hours 20 minutes of runtime

    params = [
        {'curriculum': curriculum_tuple}
    ]

    # one expert topology
    topology = SeT1Bt
    run_measurement_task1(topology.__name__, params, args, topology_class=topology)


if __name__ == '__main__':
    """Runs just the SPFlock on SEDatasetNode."""
    run_measurements_for_task1(custom_durations=True)
