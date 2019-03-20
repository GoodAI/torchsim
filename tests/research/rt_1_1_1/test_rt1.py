import pytest

from torchsim.research.experiment_templates.sp_learning_convergence_template import SpLearningConvergenceExperimentTemplate

from torchsim.research.research_topics.rt_1_1_1_one_expert_sp.adapters.se_dataset_sp_learning_convergence_adapter import \
    SeDatasetSpLearningConvergenceTopologyAdapter
from torchsim.research.research_topics.rt_1_1_1_one_expert_sp.adapters.sp_mnist_learning_convergence_adapter import \
    MnistSpLearningConvergenceTopologyAdapter
from torchsim.research.research_topics.rt_1_1_1_one_expert_sp.topologies.mnist_sp_topology import MnistSpTopology
from torchsim.research.research_topics.rt_1_1_1_one_expert_sp.topologies.se_dataset_sp_topology import SeDatasetSpTopology

# TODO: It is not deterministic, should be fixed. - added to Trello
@pytest.mark.xfail
@pytest.mark.slow
def test_rt1_mnist_sp_determinism():
    """Seeds should work on the topology level.

    Test that it is possible to collect two identical measurements.
    If the seeds are different, the results should be different.
    """

    name = "mnist-sp-determinism_test"

    # learning rate is too high, the cluster centers oscillate, but still it measures some values
    params = [
        {'dataset_seed': 0, 'model_seed': 0, 'num_cc': 2, 'lr': 0.1, 'cbt': 1, 'batch_s': 3},
        {'dataset_seed': 0, 'model_seed': 0, 'num_cc': 2, 'lr': 0.1, 'cbt': 1, 'batch_s': 3},
        # note: setting just the learned_model_seed to something else might or might not produce equal results
        # it is generally safer to change seeds of both nodes
        {'dataset_seed': 0, 'model_seed': 1123, 'num_cc': 2, 'lr': 0.1, 'cbt': 1, 'batch_s': 3},
    ]

    # empirically chosen so that the MI contains some nonzero values
    max_steps = 100

    experiment = SpLearningConvergenceExperimentTemplate(
        MnistSpLearningConvergenceTopologyAdapter(),
        MnistSpTopology,
        params,
        max_steps=max_steps,
        num_classes=10,
        measurement_period=1,
        sliding_window_size=9999,
        experiment_name=name,
        save_cache=False,
        load_cache=False,
        clear_cache=False,
        computation_only=False,
        disable_plt_show=True)

    # collects the measurements, after each run it computes mutual info of the output and labels
    experiment._collect_measurements()

    assert len(experiment._average_delta[0]) > 5

    assert experiment._average_delta[0] == experiment._average_delta[1]
    assert experiment._average_delta[0] != experiment._average_delta[2]


# TODO: It is not deterministic, should be fixed. - added to Trello
@pytest.mark.xfail
@pytest.mark.slow
def test_rt1_se_sp_determinism():
    """Seeds should work on the topology level.

    Test that it is possible to collect two identical measurements.
    If the seeds are different, the results should be different.
    """

    name = "se-sp-determinism_test"

    params = [
        {'dataset_seed': 0, 'model_seed': 0, 'num_cc': 2, 'lr': 1, 'cbt': 1, 'batch_s': 3},
        {'dataset_seed': 0, 'model_seed': 0, 'num_cc': 2, 'lr': 1, 'cbt': 1, 'batch_s': 3},
        {'dataset_seed': 125, 'model_seed': 1123, 'num_cc': 2, 'lr': 1, 'cbt': 1, 'batch_s': 3},
    ]

    # empirically chosen so that the MI contains some nonzero values
    max_steps = 100

    experiment = SpLearningConvergenceExperimentTemplate(
        SeDatasetSpLearningConvergenceTopologyAdapter(),
        SeDatasetSpTopology,
        params,
        max_steps=max_steps,
        num_classes=100,
        measurement_period=1,
        sliding_window_size=9999,  # be aware that the mutual information depends on this value
        sliding_window_stride=9999,
        experiment_name=name,
        save_cache=False,
        load_cache=False,
        clear_cache=False,
        computation_only=False,
        disable_plt_show=True)

    # collects the measurements, after each run it computes mutual info of the output and labels
    experiment._collect_measurements()

    assert len(experiment._average_delta[0]) > 5

    assert experiment._average_delta[0] == experiment._average_delta[1]
    assert experiment._average_delta[0] != experiment._average_delta[2]

