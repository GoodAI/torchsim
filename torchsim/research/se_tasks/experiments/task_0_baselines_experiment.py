import sys

from eval_utils import parse_test_args, run_just_model
from torchsim.research.se_tasks.experiments.task_0_experiment_template import Task0ExperimentTemplate
from torchsim.research.se_tasks.adapters.task_0_stats_basic_adapter import Task0StatsBasicAdapter
from torchsim.topologies.baselines_rl_topology import BaselinesReinforcementLearningTopology

DEF_MAX_STEPS = sys.maxsize


def read_max_steps(max_steps: int = None):
    if max_steps is None:
        return DEF_MAX_STEPS
    return max_steps


def run_measurement(name, params, args, topology_class, max_steps: int = None):
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
        run_just_model(BaselinesReinforcementLearningTopology(**params[0]), gui=True)
    else:
        print(f'======================================= Measuring model: {name}')
        experiment.run()


if __name__ == '__main__':
    """Runs the Reinforcement Learning Baseline on Tasks 0, 1, 2."""
    args = parse_test_args()
    params = [
        {'curriculum': (0, 1, 2, -1)}
    ]

    # one expert topology
    topology = BaselinesReinforcementLearningTopology
    run_measurement(topology.__name__, params, args, topology_class=topology)

