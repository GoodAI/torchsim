import sys

from eval_utils import parse_test_args, run_just_model
from torchsim.research.se_tasks.experiments.task_0_experiment_template import Task0ExperimentTemplate
from torchsim.research.se_tasks.adapters.task_0_stats_basic_adapter import Task0StatsBasicAdapter
from torchsim.research.se_tasks.topologies.se_task0_basic_topology import SeT0BasicTopology
from torchsim.research.se_tasks.topologies.se_task0_convolutionalSP_topology import SeT0ConvSPTopology
from torchsim.research.se_tasks.topologies.se_task0_narrow_hierarchy import SeT0NarrowHierarchy

DEF_MAX_STEPS = sys.maxsize


def read_max_steps(max_steps: int = None):
    if max_steps is None:
        return DEF_MAX_STEPS
    return max_steps


def run_measurement_task0(name, params, args, topology_class, max_steps: int = None):
    """Runs the experiment with specified params, see the parse_test_args method for arguments."""

    experiment = Task0ExperimentTemplate(
        Task0StatsBasicAdapter(),
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
        run_just_model(SeT0BasicTopology(**params[0]), gui=True)
    else:
        print(f'======================================= Measuring model: {name}')
        experiment.run()


def run_measurements_for_task0(custom_durations: bool = False):
    args = parse_test_args()

    if custom_durations:
        curriculum_tuple = (2000, -1)  # dev curriculum, 15 minutes of runtime (or your own duration)
    else:
        curriculum_tuple = (0, -1)  # evaluation curriculum, 2 hours 40 minutes of runtime

    params = [
        {'curriculum': curriculum_tuple}
    ]

    # one expert topology
    topology = SeT0BasicTopology
    run_measurement_task0(topology.__name__, params, args, topology_class=topology)

    # narrow hierarchy
    topology = SeT0NarrowHierarchy
    run_measurement_task0(topology.__name__, params, args, topology_class=topology)

    # conv SP
    topology = SeT0ConvSPTopology
    run_measurement_task0(topology.__name__, params, args, topology_class=topology)


if __name__ == '__main__':
    """Runs just the SPFlock on SEDatasetNode."""
    run_measurements_for_task0(custom_durations=True)
