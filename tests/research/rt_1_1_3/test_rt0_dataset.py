import pytest

from torchsim.research.experiment_templates.dataset_simulation_running_stats_template import \
    DatasetSeSimulationRunningStatsExperimentTemplate
from torchsim.research.experiment_templates.simulation_running_stats_template import \
    SeSimulationRunningStatsExperimentTemplate
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.adapters.se_dataset_ta_running_stats_adapter import \
    SeDatasetTaRunningStatsAdapter
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.adapters.se_ta_running_stats_adapter import \
    SeTaRunningStatsAdapter
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.topologies.se_dataset_sp_lrf import SeDatasetSpLrf
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.topologies.se_dataset_ta_lrf import SeDatasetTaLrf
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.topologies.se_ta_lrf_t0 import SeTaLrfT0


@pytest.mark.slow
def test_se_dataset_sp():
    """Just try to do several steps of the experiment."""
    max_steps = 20

    name = 'se_dataset_sp_test'
    params = [
        {'eox': 2, 'eoy': 2, 'num_cc': 10, 'batch_s': 3},
        {'eox': 2, 'eoy': 2, 'num_cc': 25, 'batch_s': 3}]

    experiment = DatasetSeSimulationRunningStatsExperimentTemplate(
        SeDatasetTaRunningStatsAdapter(),
        SeDatasetSpLrf,
        params,
        max_steps=max_steps,
        measurement_period=1,
        smoothing_window_size=3,
        save_cache=False,
        load_cache=False,
        clear_cache=False,
        experiment_name=name,
        computation_only=False,
        experiment_folder=None
    )
    experiment._collect_measurements()
    experiment._compute_experiment_statistics()


@pytest.mark.slow
def test_se_dataset_ta():
    """Just try to do several steps of the experiment."""
    max_steps = 20

    name = 'se_dataset_ta_test'

    params = [
        {'eox': 2, 'eoy': 2, 'num_cc': 10, 'batch_s': 3, 'tp_learn_period': 5},
        {'eox': 2, 'eoy': 2, 'num_cc': 25, 'batch_s': 3, 'tp_learn_period': 5}]

    experiment = DatasetSeSimulationRunningStatsExperimentTemplate(
        SeDatasetTaRunningStatsAdapter(),
        SeDatasetTaLrf,
        params,
        max_steps=max_steps,
        measurement_period=1,
        smoothing_window_size=3,
        save_cache=False,
        load_cache=False,
        clear_cache=False,
        experiment_name=name,
        computation_only=False,
        experiment_folder=None
    )

    experiment._collect_measurements()
    experiment._compute_experiment_statistics()


@pytest.mark.skip("This test would need the space to end the curriculum.")
@pytest.mark.slow
def test_se_ta():
    """Experiment using SE."""

    max_steps = 100
    name = 'se_ta_test'

    params = [
        {'eox': 2, 'eoy': 2, 'num_cc': 10},
        {'eox': 2, 'eoy': 2, 'num_cc': 25}]

    experiment = SeSimulationRunningStatsExperimentTemplate(
        SeTaRunningStatsAdapter(),
        SeTaLrfT0,
        params,
        max_steps=max_steps,
        measurement_period=1,
        smoothing_window_size=30,
        save_cache=False,
        load_cache=False,
        clear_cache=False,
        experiment_name=name,
        computation_only=False,
        experiment_folder=None
    )

    experiment._collect_measurements()
    experiment._compute_experiment_statistics()
