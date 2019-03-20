import argparse

from eval_utils import add_test_args, run_just_model, filter_params
from torchsim.research.research_topics.rt_2_1_3_conv_temporal_compression.rt_2_1_3_experiment_template import \
    Rt213ExperimentTemplate, Rt213Adapter
from torchsim.research.research_topics.rt_2_1_3_conv_temporal_compression.topologies.l3_conv_topology import \
    L3SpConvTopology, L3ConvTopology
from torchsim.significant_nodes import BallEnvironment, SEEnvironment, BallEnvironmentParams, SeEnvironmentParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_test_args(parser)
    parser.add_argument("--sp-only", action='store_true', default=False)
    args = parser.parse_args()

    if args.sp_only:
        topology_class = L3SpConvTopology
    else:
        topology_class = L3ConvTopology

    params = [{'env_class': SEEnvironment, 'l_0_rf_dims': (3, 3), 'l_1_rf_dims': (2, 2),
               'env_params': SeEnvironmentParams(shapes=[2, 3, 7])},

              {'env_class': SEEnvironment, 'l_0_rf_dims': (3, 3), 'l_1_rf_dims': (4, 4),
               'env_params': SeEnvironmentParams(shapes=[2, 3, 7])},

              {'env_class': SEEnvironment, 'l_0_rf_dims': (2, 2), 'l_1_rf_dims': (3, 3),
               'env_params': SeEnvironmentParams(shapes=[2, 3, 7])},

              {'env_class': SEEnvironment, 'l_0_rf_dims': (3, 3), 'l_1_rf_dims': (2, 2),
               'env_params': SeEnvironmentParams(shapes=[2, 3, 7, 19, 0])},

              {'env_class': BallEnvironment, 'l_0_rf_dims': (3, 3), 'l_1_rf_dims': (3, 3),
               'env_params': BallEnvironmentParams(noise_amplitude=0, env_size=(27, 27), ball_radius=5)},
              {'env_class': BallEnvironment, 'l_0_rf_dims': (3, 3), 'l_1_rf_dims': (3, 3),
               'env_params': BallEnvironmentParams(noise_amplitude=0.1, env_size=(27, 27), ball_radius=5)},

              {'env_class': BallEnvironment, 'l_0_rf_dims': (3, 3), 'l_1_rf_dims': (2, 2),
               'env_params': BallEnvironmentParams(noise_amplitude=0, env_size=(24, 24), ball_radius=4)},
              {'env_class': BallEnvironment, 'l_0_rf_dims': (3, 3), 'l_1_rf_dims': (2, 2),
               'env_params': BallEnvironmentParams(noise_amplitude=0.1, env_size=(24, 24), ball_radius=4)},

              {'env_class': BallEnvironment, 'l_0_rf_dims': (3, 3), 'l_1_rf_dims': (4, 4),
               'env_params': BallEnvironmentParams(noise_amplitude=0, env_size=(24, 24), ball_radius=4)},
              {'env_class': BallEnvironment, 'l_0_rf_dims': (3, 3), 'l_1_rf_dims': (4, 4),
               'env_params': BallEnvironmentParams(noise_amplitude=0.1, env_size=(24, 24), ball_radius=4)},

              {'env_class': BallEnvironment, 'l_0_rf_dims': (2, 2), 'l_1_rf_dims': (3, 3),
               'env_params': BallEnvironmentParams(noise_amplitude=0, env_size=(24, 24), ball_radius=4)},
              {'env_class': BallEnvironment, 'l_0_rf_dims': (2, 2), 'l_1_rf_dims': (3, 3),
               'env_params': BallEnvironmentParams(noise_amplitude=0.1, env_size=(24, 24), ball_radius=4)},


              # {'noise_amplitude': 0.1, 'env_size': (27, 27), 'l_0_rf_dims': (3, 3), 'l_1_rf_dims': (3, 3),
              #  'ball_radius': 5},
              # {'noise_amplitude': 0, 'env_size': (8, 8), 'l_0_rf_dims': (2, 2), 'l_1_rf_dims': (2, 2),
              #  'ball_radius': 2},
              # {'noise_amplitude': 0.1, 'env_size': (8, 8), 'l_0_rf_dims': (2, 2), 'l_1_rf_dims': (2, 2),
              #  'ball_radius': 2},
              # {'noise_amplitude': 0, 'env_size': (18, 18), 'l_0_rf_dims': (2, 2), 'l_1_rf_dims': (3, 3),
              #  'ball_radius': 3},
              # {'noise_amplitude': 0.1, 'env_size': (18, 18), 'l_0_rf_dims': (2, 2), 'l_1_rf_dims': (3, 3),
              #  'ball_radius': 3},
              # {'noise_amplitude': 0, 'env_size': (50, 50), 'l_0_rf_dims': (5, 5), 'l_1_rf_dims': (5, 5),
              #  'ball_radius': 6},
              # {'noise_amplitude': 0.1, 'env_size': (50, 50), 'l_0_rf_dims': (5, 5), 'l_1_rf_dims': (5, 5),
              #  'ball_radius': 6},
              # {'noise_amplitude': 0, 'env_size': (50, 50), 'l_0_rf_dims': (5, 5), 'l_1_rf_dims': (2, 2),
              #  'ball_radius': 6},
              # {'noise_amplitude': 0.1, 'env_size': (50, 50), 'l_0_rf_dims': (5, 5), 'l_1_rf_dims': (2, 2),
              #  'ball_radius': 6}
              ]

    params = filter_params(args, params)

    experiment = Rt213ExperimentTemplate(
        adapter=Rt213Adapter(),
        topology_class=topology_class,
        models_params=params,
        overall_training_steps=40000,
        num_testing_steps=1000,
        num_testing_phases=10,
        sub_experiment_name="MAIN",
        computation_only=args.computation_only,
        save_cache=args.save,
        load_cache=args.load,
        clear_cache=args.clear,
        experiment_folder=args.alternative_results_folder
    )

    if args.run_gui:
        run_just_model(topology_class(**params[-1]), gui=True, persisting_observer_system=True)
    else:
        experiment.run()
