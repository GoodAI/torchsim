
from eval_utils import run_just_model, parse_test_args
from torchsim.research.experiment_templates.dataset_simulation_running_stats_template import \
    DatasetSeSimulationRunningStatsExperimentTemplate
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.adapters.se_dataset_ta_running_stats_adapter import \
    SeDatasetTaRunningStatsAdapter
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.topologies.se_dataset_ta_lrf import SeDatasetTaLrf

DEF_MAX_STEPS = 3500


def read_max_steps(max_steps: int = None):
    if max_steps is None:
        return DEF_MAX_STEPS
    return max_steps


def run_measurement(name, params, args, max_steps: int = None):
    """Runs the experiment with specified params, see the parse_test_args method for arguments"""

    experiment = DatasetSeSimulationRunningStatsExperimentTemplate(
        SeDatasetTaRunningStatsAdapter(),
        SeDatasetTaLrf,
        params,
        max_steps=read_max_steps(max_steps),
        measurement_period=1,
        smoothing_window_size=99,
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


def run_num_cc(args, max_steps: int = None):

    name = 'FL_num_cluster_centers'

    params = [
        {'eox': 2, 'eoy': 2, 'num_cc': 10},
        {'eox': 2, 'eoy': 2, 'num_cc': 25},
        {'eox': 2, 'eoy': 2, 'num_cc': 50},
        {'eox': 2, 'eoy': 2, 'num_cc': 100},
        {'eox': 2, 'eoy': 2, 'num_cc': 250},
        {'eox': 2, 'eoy': 2, 'num_cc': 350},
        {'eox': 2, 'eoy': 2, 'num_cc': 750},  # was out of memory before optimization
        {'eox': 2, 'eoy': 2, 'num_cc': 1000},
        {'eox': 2, 'eoy': 2, 'num_cc': 2000},
        # {'eox': 2, 'eoy': 2, 'num_cc': 3500},  # this crashes on "cannot join the thread" during TP learning
    ]

    run_measurement(name, params, args, max_steps)


def run_num_experts(args, max_steps=None):
    name = 'FL_num_experts'
    params = [
        {'eox': 1, 'eoy': 1, 'num_cc': 100},
        {'eox': 2, 'eoy': 2, 'num_cc': 100},
        {'eox': 4, 'eoy': 4, 'num_cc': 100},
        {'eox': 8, 'eoy': 8, 'num_cc': 100},
        {'eox': 16, 'eoy': 16, 'num_cc': 100},
        {'eox': 32, 'eoy': 32, 'num_cc': 100},
        # {'eox': 64, 'eoy': 32, 'num_cc': 100},
        # {'eox': 64, 'eoy': 64, 'num_cc': 100},  # was out of memory before optimization
    ]
    run_measurement(name, params, args, max_steps)


def run_batch_size(args, max_steps=None):
    name = 'FL_batch_size'
    params = [
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 50},
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 300},
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 700},
        # {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 1200},
        # {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 1500},  # 1500: was out of memory before optim..
    ]
    run_measurement(name, params, args, max_steps)


def run_tp_learn_period(args, max_steps=None):
    name = 'FL_tp_learn_period'
    params = [
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 300, 'tp_learn_period': 1},
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 300, 'tp_learn_period': 10},
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 300, 'tp_learn_period': 50},
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 300, 'tp_learn_period': 100},
    ]
    run_measurement(name, params, args, max_steps)


def run_tp_max_encountered_seq(args, max_steps=None):
    name = 'FL_tp_max_enc_seq'
    params = [
        # {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 300, 'tp_learn_period': 50, 'tp_max_enc_seq': 700000},# this order crashes during the second run!
        # {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 300, 'tp_learn_period': 50, 'tp_max_enc_seq': 1000},
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 300, 'tp_learn_period': 50, 'tp_max_enc_seq': 1000},
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 300, 'tp_learn_period': 50, 'tp_max_enc_seq': 10000},
        # {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 300, 'tp_learn_period': 50, 'tp_max_enc_seq': 1000000}, # 1500000 out of mem
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 300, 'tp_learn_period': 50, 'tp_max_enc_seq': 100000},
        # {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 300, 'tp_learn_period': 50, 'tp_max_enc_seq': 500000},
        ]
    run_measurement(name, params, args, max_steps)


def run_debug(args, max_steps=None):
    name = 'FL_batch_size'
    params = [
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 50},
        {'eox': 4, 'eoy': 4, 'num_cc': 100, 'batch_s': 300}]
    run_measurement(name, params, args, max_steps)


if __name__ == '__main__':
    """
    Runs ExpertFlock on the SEDataset
    """
    arg = parse_test_args()
    ms = None

    run_debug(arg, 1500)

    # run_num_cc(arg, max_steps=ms)
    # run_num_experts(arg, max_steps=ms)
    # run_batch_size(arg, max_steps=ms)
    # run_tp_learn_period(arg, max_steps=ms)
    # run_tp_max_encountered_seq(arg, max_steps=ms)
