
from eval_utils import run_just_model, parse_test_args
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.topologies.se_dataset_sp_lrf_debug import \
    SeDatasetSpLrfDebug



def run_experiment(use_gui: bool, save: bool, load: bool, clear: bool):
    # specify the sets of params to be used in the consecutive runs inside the experiment.run()
    params = [
        {'eox': 2, 'eoy': 2, 'num_cc': 10},
        {'eox': 2, 'eoy': 2, 'num_cc': 25},
        {'eox': 2, 'eoy': 2, 'num_cc': 50},
        {'eox': 2, 'eoy': 2, 'num_cc': 100},
        {'eox': 2, 'eoy': 2, 'num_cc': 250},
        {'eox': 2, 'eoy': 2, 'num_cc': 350},
        {'eox': 2, 'eoy': 2, 'num_cc': 550},
    ]

    # NOT supported
    # experiment = SpLearningConvergenceExperimentTemplate(
    #     MnistSpLearningConvergenceTopologyAdapter(),
    #     MnistSpTopology,
    #     params,
    #     max_steps=100,
    #     measurement_period=1,
    #     evaluation_period=15)

    if use_gui:
        # run_just_model(SeDatasetSpLrfDebug(**params[0]), gui=True)
        run_just_model(SeDatasetSpLrfDebug(**params[0]), gui=True)
    else:
        # experiment.run()
        pass


if __name__ == '__main__':
    args = parse_test_args()
    run_experiment(args.run_gui, args.save, args.load, args.clear)