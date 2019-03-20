
from eval_utils import run_just_model, parse_test_args
from torchsim.research.experiment_templates.dataset_simulation_running_stats_template import \
    DatasetSeSimulationRunningStatsExperimentTemplate
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.adapters.se_dataset_ta_running_stats_adapter import \
    SeDatasetTaRunningStatsAdapter
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.topologies.se_dataset_sp_lrf import SeDatasetSpLrf
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.topologies.se_dataset_ta_lrf import SeDatasetTaLrf

DEF_MAX_STEPS = 1000


def read_max_steps(max_steps: int = None):
    if max_steps is None:
        return DEF_MAX_STEPS
    return max_steps


def run_measurement(name, params, args, max_steps: int = None):
    """Runs the experiment with specified params, see the parse_test_args method for arguments"""

    experiment = DatasetSeSimulationRunningStatsExperimentTemplate(
        SeDatasetTaRunningStatsAdapter(),
        SeDatasetSpLrf,
        params,
        max_steps=read_max_steps(max_steps),
        measurement_period=1,
        smoothing_window_size=30,
        save_cache=args.save,
        load_cache=args.load,
        clear_cache=args.clear,
        experiment_name=name,
        computation_only=args.computation_only,
        experiment_folder=args.alternative_results_folder
    )

    if args.run_gui:
        run_just_model(SeDatasetTaLrf(**params[0]), gui=True)
    else:
        print(f'======================================= Measuring model: {name}')
        experiment.run()


def run_num_cc(args, max_steps):
    name = 'num_cluster_centers'
    params = [
        {'eox': 2, 'eoy': 2, 'num_cc': 10},
        {'eox': 2, 'eoy': 2, 'num_cc': 25},
        {'eox': 2, 'eoy': 2, 'num_cc': 50},
        {'eox': 2, 'eoy': 2, 'num_cc': 100},
        {'eox': 2, 'eoy': 2, 'num_cc': 250},
        {'eox': 2, 'eoy': 2, 'num_cc': 350},
        # {'eox': 2, 'eoy': 2, 'num_cc': 550}, # sometimes out of memory :(
        # {'experts_on_x': 2, 'experts_on_y': 2, 'num_cc': 750},
        # {'experts_on_x': 2, 'experts_on_y': 2, 'num_cc': 950}, # sometimes works
    ]

    run_measurement(name, params, args, max_steps)


def run_num_experts(args, max_steps):
    name = 'num_experts'
    params = [
        {'eox': 1, 'eoy': 1, 'num_cc': 100},
        # {'eox': 2, 'eoy': 1, 'num_cc': 100},
        {'eox': 2, 'eoy': 2, 'num_cc': 100},
        # {'eox': 4, 'eoy': 2, 'num_cc': 100},
        {'eox': 4, 'eoy': 4, 'num_cc': 100},
        # {'eox': 8, 'eoy': 4, 'num_cc': 100},
        {'eox': 8, 'eoy': 8, 'num_cc': 100},
        # {'eox': 16, 'eoy': 8, 'num_cc': 100},
        {'eox': 16, 'eoy': 16, 'num_cc': 100},
        # {'eox': 32, 'eoy': 16, 'num_cc': 100},
        {'eox': 32, 'eoy': 32, 'num_cc': 100},
        # {'eox': 64, 'eoy': 32, 'num_cc': 100},
        {'eox': 64, 'eoy': 64, 'num_cc': 100},
    ]
    run_measurement(name, params, args, max_steps)


def run_batch_size(args, max_steps):
    name = 'batch_size'
    params = [
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 50},
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 150},
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 300},
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 450},
        # {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 700},
        # {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 1000}, # 1200 out of memory for multiple experiments
        # {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 1500},  # 1500: out of memory
    ]
    run_measurement(name, params, args, max_steps)


if __name__ == '__main__':
    """
    Runs just the SPFlock on SEDatasetNode
    """
    arg = parse_test_args()
    ms = None

    run_num_cc(arg, ms)                     # was 400 steps
    run_num_experts(arg, ms)                # was 400 steps
    run_batch_size(arg, max_steps=1500)     # 1500 steps

