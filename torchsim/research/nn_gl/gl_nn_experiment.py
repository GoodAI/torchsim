import argparse
import logging
from typing import Dict, Any

import numpy as np

from eval_utils import parse_test_args, run_experiment, add_test_args
from torchsim.core.eval2.experiment import Experiment
from torchsim.core.eval2.experiment_runner_params import ExperimentParams
from torchsim.core.eval2.experiment_template_base import TopologyFactory
from torchsim.core.utils.sequence_generator import SequenceGenerator, diagonal_transition_matrix
from torchsim.research.nn_gl.gl_nn_experiment_template import GradualLearningExperimentTemplate, \
    GradualLearningExperimentTemplateParams
from torchsim.topologies.gl_nn_topology import GlNnTopology

logger = logging.getLogger(__name__)


class GlNnTopologyFactory(TopologyFactory[GlNnTopology]):
    def get_default_parameters(self) -> Dict[str, Any]:
        return {}

    def create_topology(self, **kwargs) -> GlNnTopology:
        return GlNnTopology(**kwargs)


def run_measurement(args, topologies_params, template_params):
    topology_factory = GlNnTopologyFactory()

    template = GradualLearningExperimentTemplate(template_params)

    runner_parameters = ExperimentParams(max_steps=0,
                                         save_cache=args.save,
                                         load_cache=args.load,
                                         clear_cache=args.clear,
                                         calculate_statistics=not args.computation_only,
                                         experiment_folder=args.alternative_results_folder)

    experiment = Experiment(template, topology_factory, topologies_params,
                            runner_parameters)

    # run_experiment_with_ui(experiment)
    run_experiment(experiment)


if __name__ == '__main__':
    all_sequences = diagonal_transition_matrix(4, 0.8)
    third_sequence_excluded = np.array(
        [
            [0.8, 0.1, 0.0, 0.1],
            [0.1, 0.8, 0.0, 0.1],
            [1.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0, 0.8]
        ])
    only_third_sequence = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
    fourth_sequence_excluded = np.array(
        [
            [0.8, 0.1, 0.1, 0.0],
            [0.1, 0.8, 0.1, 0.0],
            [0.1, 0.1, 0.8, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ])
    only_fourth_sequence = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    all_seq = SequenceGenerator(
        [
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
            [2, 1, 3, 2, 1, 3, 2, 1, 3],
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
            [5, 4, 3, 5, 4, 3, 5, 4, 3],
        ]
        , all_sequences)
    no_fourth = SequenceGenerator(
        [
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
            [2, 1, 3, 2, 1, 3, 2, 1, 3],
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
            [1]
        ]
        , fourth_sequence_excluded)
    only_fourth = SequenceGenerator(
        [
            [1],
            [2],
            [3],
            [5, 4, 3, 5, 4, 3, 5, 4, 3],
        ]
        , only_fourth_sequence)
    no_third = SequenceGenerator(
        [
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
            [2, 1, 3, 2, 1, 3, 2, 1, 3],
            [1],
            [5, 4, 3, 5, 4, 3, 5, 4, 3],
        ]
        , third_sequence_excluded)
    only_third = SequenceGenerator(
        [
            [1],
            [2],
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
            [5],
        ]
        , only_third_sequence)
    no_seq = SequenceGenerator(
        [
            [1]
        ]
        , np.ones(1))

    parser = argparse.ArgumentParser()
    add_test_args(parser)
    parser.add_argument("--experiment", type=str, help="not-forgetting or adding_new_label or convergence or "
                                                       "adding_new_behavior", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    experiment = args.experiment
    experiment = experiment.replace('-', '_')

    # sequence order:
    # 0 training_configuration
    # 1 testing_configuration
    # 2 untraining_configuration
    # 3 retraining_configuration

    if experiment == "not_forgetting":
        sequences_not_forgetting = [all_seq.clone(), only_fourth.clone(), no_fourth.clone(), all_seq.clone()]
        sequences_not_forgetting_baseline = [no_fourth, only_fourth, no_fourth.clone(), all_seq]
        params = GradualLearningExperimentTemplateParams(untraining_steps=30000, testing_steps=200,
                                                         retraining_steps=200, retraining_phases=20,
                                                         every_second_is_baseline=True,
                                                         experiment_name=experiment)
        num_predictors = 15
        topologies_params = [{"sequence_generators": sequences_not_forgetting,
                              "num_predictors": num_predictors},
                             {"sequence_generators": sequences_not_forgetting_baseline,
                              "num_predictors": num_predictors}]

    elif experiment == "adding_new_label":
        sequences = [no_third.clone(), only_third.clone(), no_seq.clone(), all_seq.clone()]
        params = GradualLearningExperimentTemplateParams(untraining_steps=0, testing_steps=100,
                                                         retraining_steps=100, retraining_phases=10,
                                                         experiment_name=experiment)
        num_predictors = 3
        topologies_params = [{"sequence_generators": sequences, "num_predictors": num_predictors}] * 5

    elif experiment == "adding_new_behavior":
        sequences = [no_fourth.clone(), only_fourth.clone(), no_seq.clone(), all_seq.clone()]
        params = GradualLearningExperimentTemplateParams(untraining_steps=0, testing_steps=1000,
                                                         retraining_steps=1000, retraining_phases=40,
                                                         experiment_name=experiment)
        num_predictors = 3
        topologies_params = [{"sequence_generators": sequences, "num_predictors": num_predictors}]

    elif experiment == "convergence":
        sequences = [all_seq.clone(), no_seq.clone(), no_seq.clone(), no_seq.clone()]
        params = GradualLearningExperimentTemplateParams(initial_training_steps=10000,
                                                         untraining_steps=0, testing_steps=0,
                                                         retraining_steps=0, retraining_phases=0,
                                                         experiment_name=experiment)
        topologies_params = [{"sequence_generators": sequences, "num_predictors": num_predictors}
                             for num_predictors in [3] * 5]

    else:
        raise NotImplemented()

    if args.debug:
        params = GradualLearningExperimentTemplateParams(initial_training_steps=500,
                                                         untraining_steps=50, testing_steps=50,
                                                         retraining_steps=50, retraining_phases=3,
                                                         experiment_name="debug")

    run_measurement(args, topologies_params, params)
