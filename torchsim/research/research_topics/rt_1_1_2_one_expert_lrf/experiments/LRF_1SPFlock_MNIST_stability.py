import argparse

from eval_utils import run_just_model
from torchsim.research.experiment_templates.lrf_1sp_flock_template import Lrf1SpFlockExperimentTemplate

from torchsim.research.research_topics.rt_1_1_2_one_expert_lrf.adapters.LRF_1SPFlock_MNIST import Lrf1SpFlockMnistTemplate
from torchsim.research.research_topics.rt_1_1_2_one_expert_lrf.topologies.lrf_topology import LrfTopology


def run_experiment(run_gui: bool, save: bool, load: bool, clear: bool):
    params = [
        {"expert_width": 28, "n_cluster_centers": 100},
        {"expert_width": 28, "n_cluster_centers": 100},
        {"expert_width": 28, "n_cluster_centers": 100},
        {"expert_width": 28, "n_cluster_centers": 100},
        {"expert_width": 28, "n_cluster_centers": 100},
        {"expert_width": 28, "n_cluster_centers": 100},
        {"expert_width": 28, "n_cluster_centers": 100},
        {"expert_width": 28, "n_cluster_centers": 100},
    ]

    for i, param in enumerate(params):
        param['seed'] = i
        param['training_phase_steps'] = 500
        param['testing_phase_steps'] = 1000

    experiment = Lrf1SpFlockExperimentTemplate(
        Lrf1SpFlockMnistTemplate(),
        LrfTopology,
        params,
        max_steps=15000,
        measurement_period=1,
        save_cache=save,
        load_cache=load,
        clear_cache=clear
    )

    if run_gui:
        run_just_model(LrfTopology(**params[0]), gui=True)
    else:
        experiment.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-gui", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--clear", action="store_true", default=False)
    args = parser.parse_args()
    run_experiment(args.run_gui, args.save, args.load, args.clear)
