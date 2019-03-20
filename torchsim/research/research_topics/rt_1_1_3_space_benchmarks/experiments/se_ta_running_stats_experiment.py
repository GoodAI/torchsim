
from eval_utils import run_just_model, parse_test_args
from torchsim.research.experiment_templates.simulation_running_stats_template import SeSimulationRunningStatsExperimentTemplate
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.adapters.se_ta_running_stats_adapter import \
    SeTaRunningStatsAdapter
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.topologies.se_ta_lrf_t0 import SeTaLrfT0

DEF_MAX_STEPS = 2000


def read_max_steps(max_steps: int = None):
    if max_steps is None:
        return DEF_MAX_STEPS
    return max_steps


def run_measurement(name, params, args, max_steps: int = None):
    """Runs the experiment with specified params, see the parse_test_args method for arguments"""

    experiment = SeSimulationRunningStatsExperimentTemplate(
        SeTaRunningStatsAdapter(),
        SeTaLrfT0,
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
        run_just_model(SeTaLrfT0(**params[0]), gui=True)
    else:
        print(f'======================================= Measuring model: {name}')
        experiment.run()


def run_num_cc(args, max_steps):
    name = 'FL_SE_num_cluster_centers'
    params = [
        # {'eox': 2, 'eoy': 2, 'num_cc': 10},
        # {'eox': 2, 'eoy': 2, 'num_cc': 25},
        {'eox': 2, 'eoy': 2, 'num_cc': 50},
        # {'eox': 2, 'eoy': 2, 'num_cc': 100},
        # {'eox': 2, 'eoy': 2, 'num_cc': 250},
        {'eox': 2, 'eoy': 2, 'num_cc': 350},
    ]

    run_measurement(name, params, args, max_steps)


def run_num_experts(args, max_steps):
    name = 'FL_SE_num_experts'
    params = [
        {'eox': 1, 'eoy': 1, 'num_cc': 100},
        {'eox': 2, 'eoy': 2, 'num_cc': 100},
        {'eox': 4, 'eoy': 4, 'num_cc': 100},
        # {'eox': 8, 'eoy': 8, 'num_cc': 100},
        {'eox': 16, 'eoy': 16, 'num_cc': 100},
        # {'eox': 32, 'eoy': 32, 'num_cc': 100},
        # {'eox': 64, 'eoy': 32, 'num_cc': 100},
    ]
    run_measurement(name, params, args, max_steps)


def run_skip_frames(args, max_steps):
    """Different numbers of frames skipped on the side of SE"""
    name = 'FL_SE_skip_frames'
    params = [
        {'se_skip_frames': 1},
        {'se_skip_frames': 9},
        {'se_skip_frames': 20},
        {'se_skip_frames': 30},
        {'se_skip_frames': 70}
    ]
    run_measurement(name, params, args, max_steps)


def run_skip_frames_more_experts(args, max_steps):
    name = 'FL_SE_skip_frames_more_experts'
    params = [
        {'eox': 16, 'eoy': 16, 'se_skip_frames': 1},
        {'eox': 16, 'eoy': 16, 'se_skip_frames': 9},
        {'eox': 16, 'eoy': 16, 'se_skip_frames': 20},
        {'eox': 16, 'eoy': 16, 'se_skip_frames': 30},
        {'eox': 16, 'eoy': 16, 'se_skip_frames': 70}
    ]
    run_measurement(name, params, args, max_steps)


if __name__ == '__main__':
    """
    Runs ExpertFlock on the SETask0, launch the SE with Task0 duration set to big number:
    se_school/se_school_mod/SchoolMod/School/Tasks/Task0.cs: TRAINING_PHASE_TIME
    """
    arg = parse_test_args()
    ms = None

    run_num_cc(arg, ms)                     # was 2000 steps
    run_num_experts(arg, ms)                # was 2000
    run_skip_frames(arg, ms)                # was 2000
    run_skip_frames_more_experts(arg, ms)   # was 2000

