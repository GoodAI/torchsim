import argparse
import itertools

from eval_utils import run_just_model
from torchsim.research.experiment_templates.lrf_1sp_flock_template import Lrf1SpFlockExperimentTemplate
from torchsim.research.research_topics.rt_1_1_2_one_expert_lrf.adapters.LRF_1SPFlock_MNIST import Lrf1SpFlockMnistTemplate
from torchsim.research.research_topics.rt_1_1_2_one_expert_lrf.topologies.lrf_topology import LrfTopology


def run_experiment(run_gui: bool, save: bool, load: bool, clear: bool, computation_only: bool,
                   alternative_results_folder: str, convolutional: bool):
    expert_widths = [14, 7, 4, 2, 1]
    n_cluster_centers = [500, 100, 10, 5, 2]
    strides = range(1, 14)

    square_side = 28
    params = [
        dict(
            [
                ("expert_width", ew),
                ("n_cluster_centers", ncc),
                ("stride", stride)
            ]
        )
        for ew, ncc, stride in itertools.product(
            expert_widths,
            n_cluster_centers,
            strides
        )
        if (square_side - ew) % stride == 0 and stride <= ew and ew + stride <= square_side and
           ((square_side - (ew - stride)) // stride) * ncc < 1000
    ]

    params.append({"expert_width": square_side, "n_cluster_centers": 2, "stride": square_side})
    params.append({"expert_width": square_side, "n_cluster_centers": 5, "stride": square_side})
    params.append({"expert_width": square_side, "n_cluster_centers": 10, "stride": square_side})
    params.append({"expert_width": square_side, "n_cluster_centers": 100, "stride": square_side})
    params.append({"expert_width": square_side, "n_cluster_centers": 500, "stride": square_side})

    for i, param in enumerate(params):
        param['is_convolutional'] = convolutional
        param['seed'] = 0
        param['training_phase_steps'] = 200
        param['testing_phase_steps'] = 800

    params.sort(key=lambda x: x["n_cluster_centers"] / x["stride"] * x['expert_width'], reverse=True)

    # params = params[::-1]

    print('Parameters: _____________________________')
    for param in params:
        print(param)
    print(f"total {len(params)}")
    print('_________________________________________')

    experiment = Lrf1SpFlockExperimentTemplate(
        Lrf1SpFlockMnistTemplate(),
        LrfTopology,
        params,
        max_steps=30000,
        measurement_period=1,
        save_cache=save,
        load_cache=load,
        clear_cache=clear,
        computation_only=computation_only,
        experiment_folder=alternative_results_folder
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
    parser.add_argument("--computation-only", action="store_true", default=False)
    parser.add_argument("--alternative-results-folder", default=None)
    parser.add_argument("--convolutional", action="store_true", default=False)
    args = parser.parse_args()

    run_experiment(args.run_gui, args.save, args.load, args.clear, args.computation_only,
                   args.alternative_results_folder, args.convolutional)
