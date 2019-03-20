from eval_utils import run_just_model, parse_test_args
from torchsim.research.baselines.adapters.task_0_baselines_adapter import Task0BaselinesStatsBasicAdapter

from torchsim.research.se_tasks.experiments.task_0_experiment_template import Task0ExperimentTemplate
from torchsim.topologies.nnet_topology import NNetTopology

DEF_MAX_STEPS = 50000000


def read_max_steps(max_steps: int = None):
    if max_steps is None:
        return DEF_MAX_STEPS
    return max_steps


def run_measurement(name, params, args, max_steps: int = None):
    """Runs the experiment with specified params, see the parse_test_args method for arguments."""

    experiment = Task0ExperimentTemplate(
        Task0BaselinesStatsBasicAdapter(),
        NNetTopology,
        params,
        max_steps=read_max_steps(max_steps),
        measurement_period=4,
        smoothing_window_size=9,
        save_cache=args.save,
        load_cache=args.load,
        clear_cache=args.clear,
        experiment_name=name,
        computation_only=args.computation_only,
        experiment_folder=args.alternative_results_folder,
        disable_plt_show=True
    )

    if args.run_gui:
        run_just_model(NNetTopology(**params[0]), gui=True)
    else:
        print(f'======================================= Measuring model: {name}')
        experiment.run()


def run_t0_full_experiment(args, max_steps):
    name = "Task 0 - full"
    params = [{'curriculum': (0, -1)}]
    run_measurement(name, params, args, max_steps)


def run_t0_medium_experiment(args, max_steps):
    name = "Task 0 - medium"
    params = [{'curriculum': (2000, -1)}]
    run_measurement(name, params, args, max_steps)


def run_t0_fast_experiment(args, max_steps):
    name = "Task 0 - fast"
    params = [{'curriculum': (3000, -1)}]
    run_measurement(name, params, args, max_steps)


if __name__ == '__main__':
    arg = parse_test_args()

    # num_steps = 100000
    # num_steps = 3000
    # num_steps = 1000
    # num_steps = 800
    # num_steps = 500
    num_steps = None

    # run_t0_full_experiment(arg, num_steps)
    # run_t0_medium_experiment(arg, num_steps)
    run_t0_fast_experiment(arg, num_steps)
