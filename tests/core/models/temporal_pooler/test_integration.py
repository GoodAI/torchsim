import numpy as np
import pytest

import torch

from torchsim.core import get_float
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.expert_params import ExpertParams, NUMBER_OF_CONTEXT_TYPES
from torchsim.core.models.temporal_pooler import TPFlock, TrainedForwardProcessFactory
from tests.core.models.integration_test_utils import randomize_subflock, calculate_expected_results, \
    check_integration_results


def get_subflock_integration_testing_flock(params, subflock_size, device):
    flock = TPFlock(params, AllocatingCreator(device))

    flock.buffer.current_ptr = torch.randint(0, params.temporal.buffer_size, flock.buffer.current_ptr.size(),
                                             dtype=torch.int64, device=device)
    flock.buffer.total_data_written = torch.randint(0, params.spatial.buffer_size,
                                                    flock.buffer.current_ptr.size(), dtype=torch.int64, device=device)

    flock.buffer.clusters.stored_data.random_()
    flock.buffer.contexts.stored_data.random_()
    flock.buffer.outputs.stored_data.random_()
    flock.buffer.seq_probs.stored_data.random_()

    flock.action_rewards.random_()
    flock.projection_outputs.random_()
    flock.frequent_seq_occurrences.random_()
    flock.frequent_seqs.random_()
    flock.all_encountered_seqs.random_()
    flock.all_encountered_seq_occurrences.random_()
    flock.all_encountered_context_occurrences.random_()
    flock.frequent_context_likelihoods.random_()
    flock.execution_counter_forward.random_()

    # Create a subflock and populate it
    indices_np = sorted(np.random.choice(params.flock_size, subflock_size, replace=False))
    indices = torch.tensor(list(map(int, indices_np)), dtype=torch.int64, device=device).unsqueeze(dim=1)

    return flock, indices, indices_np


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
def test_forward_subflock_integration(device):
    flock_size = 10
    subflock_size = 6
    n_cluster_centers = 4
    context_size = n_cluster_centers
    buffer_size = 7
    float_dtype = get_float(device)
    n_providers = 1

    params = ExpertParams()
    params.flock_size = flock_size
    params.n_cluster_centers = n_cluster_centers

    params.temporal.n_providers = n_providers
    params.temporal.incoming_context_size = context_size
    params.temporal.buffer_size = buffer_size
    params.temporal.batch_size = buffer_size

    flock, indices, indices_np = get_subflock_integration_testing_flock(params, subflock_size, device)

    cluster_data = torch.rand((flock_size, n_cluster_centers), dtype=float_dtype, device=device)
    context_data = torch.rand((flock_size, n_providers, NUMBER_OF_CONTEXT_TYPES, n_cluster_centers), dtype=float_dtype, device=device)
    reward_data = torch.rand((flock_size, 2), dtype=float_dtype, device=device)

    forward_factory = TrainedForwardProcessFactory()
    forward = forward_factory.create(flock, cluster_data, context_data, reward_data, indices, device)
    # TODO (Test): add other tensors from the process, check this also for the untrained_forward_process
    should_update = [
        (flock.projection_outputs, forward._projection_outputs),
        (flock.action_rewards, forward._action_rewards),
        (flock.action_outputs, forward._action_outputs),
    ]

    should_not_update = [
        (flock.frequent_seqs, forward._frequent_seqs),
        (flock.frequent_seq_occurrences, forward._frequent_seq_occurrences),
        (flock.frequent_context_likelihoods, forward._frequent_context_likelihoods),
    ]

    randomize_subflock(should_update, should_not_update)

    expected_results = calculate_expected_results(should_update, should_not_update, flock_size, indices_np)

    forward.integrate()

    check_integration_results(expected_results, should_update, should_not_update)


def test_learning_subflock_integration():
    flock_size = 10
    subflock_size = 6
    n_cluster_centers = 4
    context_size = n_cluster_centers
    buffer_size = 7
    device = 'cpu'

    params = ExpertParams()
    params.flock_size = flock_size
    params.n_cluster_centers = n_cluster_centers

    params.temporal.incoming_context_size = context_size
    params.temporal.buffer_size = buffer_size
    params.temporal.batch_size = buffer_size

    flock, indices, indices_np = get_subflock_integration_testing_flock(params, subflock_size, device)

    forward = flock._create_learning_process(indices)

    should_update = [
        (flock.all_encountered_seqs, forward._all_encountered_seqs),
        (flock.all_encountered_seq_occurrences, forward._all_encountered_seq_occurrences),
        (flock.all_encountered_context_occurrences, forward._all_encountered_context_occurrences),
        (flock.frequent_seqs, forward._frequent_seqs),
        (flock.frequent_seq_occurrences, forward._frequent_seq_occurrences),
        (flock.frequent_context_likelihoods, forward._frequent_context_likelihoods),
    ]

    should_not_update = [
        (flock.buffer.seq_probs.stored_data, forward._buffer.seq_probs.stored_data),
        (flock.buffer.outputs.stored_data, forward._buffer.outputs.stored_data),
        (flock.buffer.contexts.stored_data, forward._buffer.contexts.stored_data),
        (flock.buffer.clusters.stored_data, forward._buffer.clusters.stored_data),
        (flock.buffer.current_ptr, forward._buffer.current_ptr),
        (flock.buffer.total_data_written, forward._buffer.total_data_written),
    ]

    randomize_subflock(should_update, should_not_update)

    expected_results = calculate_expected_results(should_update, should_not_update, flock_size, indices_np)

    forward.integrate()

    check_integration_results(expected_results, should_update, should_not_update)
