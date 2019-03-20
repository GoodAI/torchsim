import pytest

from torchsim.research.experiment_templates.lrf_1sp_flock_template import Lrf1SpFlockExperimentTemplate
from torchsim.research.research_topics.rt_1_1_2_one_expert_lrf.adapters.LRF_1SPFlock_MNIST import \
    Lrf1SpFlockMnistTemplate
from torchsim.research.research_topics.rt_1_1_2_one_expert_lrf.adapters.LRF_1SPFlock_SE_NAV import \
    Lrf1SpFlockSeNavTemplate
from torchsim.research.research_topics.rt_1_1_2_one_expert_lrf.topologies.lrf_topology import LrfTopology
from torchsim.research.research_topics.rt_1_1_2_one_expert_lrf.topologies.se_nav_lrf_topology import SeNavLrfTopology


@pytest.mark.slow
def test_rt2_experiment_mnist():
    params = [
        {"expert_width": 14, "n_cluster_centers": 10, "stride": 2, "training_phase_steps": 1, "testing_phase_steps": 4}
    ]

    experiment = Lrf1SpFlockExperimentTemplate(
        Lrf1SpFlockMnistTemplate(),
        LrfTopology,
        params,
        max_steps=5,
        measurement_period=1,
        save_cache=False,
        load_cache=False,
        clear_cache=False,
        experiment_folder=None
    )

    experiment._collect_measurements()
    experiment._compute_experiment_statistics()


@pytest.mark.slow
def test_rt2_experiment_mnist_stability():
    params = [
        {"expert_width": 28, "n_cluster_centers": 100}
    ]

    experiment = Lrf1SpFlockExperimentTemplate(
        Lrf1SpFlockMnistTemplate(),
        LrfTopology,
        params,
        max_steps=1,
        measurement_period=1,
        save_cache=False,
        load_cache=False,
        clear_cache=False,
        experiment_folder=None
    )

    experiment._collect_measurements()
    experiment._compute_experiment_statistics()


@pytest.mark.slow
def test_rt2_experiment_se_nav():
    params = [
        {"expert_width": 32, "n_cluster_centers": 2, "stride": 32}
    ]

    experiment = Lrf1SpFlockExperimentTemplate(
        Lrf1SpFlockSeNavTemplate(),
        SeNavLrfTopology,
        params,
        max_steps=5,
        measurement_period=1,
        save_cache=False,
        load_cache=False,
        clear_cache=False,
        computation_only=False,
        experiment_folder=None
    )

    experiment._collect_measurements()
    experiment._compute_experiment_statistics()
