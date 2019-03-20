import logging
from dataclasses import dataclass
from typing import List

import pytest

from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.eval2.experiment import Experiment
from torchsim.core.eval2.experiment_controller import TrainTestComponentParams
from torchsim.core.eval2.experiment_runner_params import ExperimentParams
from torchsim.core.eval2.scaffolding import TopologyScaffoldingFactory
from torchsim.core.experiment_runner import BasicExperimentRunner
from torchsim.core.nodes import DatasetSeObjectsNode
from torchsim.research.experiment_templates2.task0_ta_analysis_template import Task0TaAnalysisTemplate, Task0TaAnalysisParams
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.se_node_group import SeNodeGroup
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.node_groups.dummy_model_group import DummyModelGroup
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.topologies.task0_ta_analysis_topology import \
    Task0TaAnalysisTopology

logger = logging.getLogger(__name__)


@dataclass
class Params:
    experiment_params: Task0TaAnalysisParams
    train_test_params: TrainTestComponentParams


debug_params = Params(Task0TaAnalysisParams(measurement_period=1,
                                            num_classes=DatasetSeObjectsNode.label_size(),
                                            num_layers=2,
                                            sp_evaluation_period=2,
                                            show_conv_agreements=False),
                      TrainTestComponentParams(num_testing_phases=3,
                                               num_testing_steps=7,
                                               overall_training_steps=10))


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
def test_layer_wise_stats():
    """
    Run a complete measurement from the used template.
    The model is a dummy group, which computes some specific values.
    This test validates that:
        -the same values are measured and stored (e.g. boosting durations, clustering agreements..)
        -a reasonable statistics are computed from them (e.g. weak classifier accuracy from random noise)
    """
    name = "test_layer_wise_stats"

    cf_easy = None
    size = SeDatasetSize.SIZE_24

    params = [
        {
            'se_group': {'class_filter': cf_easy,
                         'image_size': size},
            'model': {}
        },
        # could be used for testing multiple runs
        # {
        #     'se_group': {'class_filter': cf_easy,
        #                  'image_size': size},
        #     'model': {}
        # }
    ]
    exp_pars = debug_params
    exp_pars.experiment_params.num_layers = 2

    scaffolding = TopologyScaffoldingFactory(Task0TaAnalysisTopology,
                                             se_group=SeNodeGroup,
                                             model=DummyModelGroup)  # a specific model for testing

    template = Task0TaAnalysisTemplate("Task 0 layer-wise stats and classification accuracy",
                                       exp_pars.experiment_params,
                                       exp_pars.train_test_params)

    runner_parameters = ExperimentParams(max_steps=exp_pars.train_test_params.max_steps,
                                         save_cache=False,
                                         load_cache=False,
                                         clear_cache=False,
                                         calculate_statistics=True)

    experiment = Experiment(template, scaffolding, params, runner_parameters)

    logger.info(f'Running model: {name}')
    experiment.run(BasicExperimentRunner())

    print('run finished, collecting the measurements')

    measurement_manager = experiment.measurement_manager

    # num boosted clusters for both layers
    num_boosted_clusters_0 = measurement_manager.get_values_from_all_runs('num_boosted_getter_' + str(0))
    num_boosted_clusters_1 = measurement_manager.get_values_from_all_runs('num_boosted_getter_' + str(1))

    def compute_num_boosted_clusters(layer_id: int, constant: float, step: int):
        return 1/(constant + step + layer_id)
    check_layer_outputs(0, 0.3, num_boosted_clusters_0, compute_num_boosted_clusters, 1)
    check_layer_outputs(1, 0.3, num_boosted_clusters_1, compute_num_boosted_clusters, 1)

    # average boosting duration
    average_boosting_dur_0 = measurement_manager.get_values_from_all_runs('average_boosting_dur_' + str(0))
    average_boosting_dur_1 = measurement_manager.get_values_from_all_runs('average_boosting_dur_' + str(1))

    def compute_avb(layer_id: int, constant: float, step: int):
        return constant + step + layer_id
    check_layer_outputs(0, 0.2, average_boosting_dur_0, compute_avb, 1)
    check_layer_outputs(1, 0.2, average_boosting_dur_1, compute_avb, 1)

    # average deltas
    average_deltas_0 = measurement_manager.get_values_from_all_runs('average_delta_' + str(0))
    average_deltas_1 = measurement_manager.get_values_from_all_runs('average_delta_' + str(1))

    def compute_avd(layer_id: int, constant: float, step: int):
        return constant + step + layer_id
    check_layer_outputs(0, 0.1, average_deltas_0, compute_avd, 1)
    check_layer_outputs(1, 0.1, average_deltas_1, compute_avd, 1)

    # MSE
    predicted_labels_mse = measurement_manager.get_custom_data_from_all_runs('predicted_labels_mse')
    baseline_labels_mse = measurement_manager.get_custom_data_from_all_runs('baseline_labels_mse')
    check_series_similar(predicted_labels_mse, baseline_labels_mse, abs_diff=0.1)

    # step-wise accuracy
    model_accuracy = measurement_manager.get_custom_data_from_all_runs('model_accuracy')
    baseline_accuracy = measurement_manager.get_custom_data_from_all_runs('baseline_accuracy')
    check_series_similar(model_accuracy, baseline_accuracy, abs_diff=0.3)

    # accuracies
    weak_class_accuracy_0 = measurement_manager.get_custom_data_from_all_runs('weak_class_accuracy_' + str(0))
    weak_class_accuracy_1 = measurement_manager.get_custom_data_from_all_runs('weak_class_accuracy_' + str(1))
    check_series_around_constant(weak_class_accuracy_0, 0.99, 0.3)
    check_series_around_constant(weak_class_accuracy_1, 0.99, 0.3)

    # cluster agreements, some of them should be 1 (all on diagonal and some that are arficially made so)
    clustering_agreements_0 = measurement_manager.get_custom_data_from_all_runs('clustering_agreements_' + str(0))
    clustering_agreements_1 = measurement_manager.get_custom_data_from_all_runs('clustering_agreements_' + str(1))

    for run_agreement in clustering_agreements_0:
        check_value_on_diagonal(run_agreement, 1.0)
    for run_agreement in clustering_agreements_1:
        check_value_on_diagonal(run_agreement, 1.0)

    # here, we check only agreements from the first run, the layer may have flock_size>1

    # layer 0: the clustering agreement between the first and second phase should be 1 (produced identical sequences)
    for flock_id in range(len(clustering_agreements_0[0])):
        assert clustering_agreements_0[0][flock_id][0][1] == 1

    # layer 1: the clustering agreement between the first and third phase should be 1
    for flock_id in range(len(clustering_agreements_1[0])):
        assert clustering_agreements_1[0][flock_id][0][2]


def check_value_on_diagonal(clustering_agreements: List[List[List[float]]], value: float):
    for run_data in clustering_agreements:
        for idx_source, run_data_source in enumerate(run_data):
            for idx_target, run_data_target in enumerate(run_data_source):
                if idx_source == idx_target:
                    assert run_data_target == value


def check_series_around_constant(data_a: List[List[float]], constant: float, abs_diff: float):
    for data_a_from_run in data_a:
        for sample_a in data_a_from_run:
            assert abs(sample_a - constant) < abs_diff


def check_series_similar(data_a: List[List[float]], data_b: List[List[float]], abs_diff: float):
    for data_a_from_run, data_b_from_run in zip(data_a, data_b):
        for sample_a, sample_b in zip(data_a_from_run, data_b_from_run):
            assert abs(sample_a - sample_b) < abs_diff


def check_layer_outputs(layer_id: int, constant: float, data: List[List[float]], value_computation, eval_period: int):

    for data_from_run in data:
        step = 1
        for sample in data_from_run:
            value = value_computation(layer_id, constant, step)
            assert sample == value
            step += eval_period


