import pytest
import torch

from torchsim.core import get_float
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.expert_params import ExpertParams, NUMBER_OF_CONTEXT_TYPES
from torchsim.core.models.flock.expert_flock import ExpertFlock
from torchsim.core.utils.tensor_utils import same


def prepare_flock_for_context(flock_size=5, input_size=3, n_cluster_centers=3, seq_lookahead=1, seq_length=3,
                              max_provider_context_size=1, n_providers=2, device='cuda', compute_reconstruction=False):
    params = ExpertParams()
    params.compute_reconstruction = compute_reconstruction
    params.flock_size = flock_size
    params.n_cluster_centers = n_cluster_centers
    params.spatial.input_size = input_size
    params.temporal.seq_lookahead = seq_lookahead
    params.temporal.seq_length = seq_length
    params.temporal.incoming_context_size = max_provider_context_size
    params.temporal.n_providers = n_providers

    flock = ExpertFlock(params, AllocatingCreator(device))

    return flock


@pytest.mark.parametrize('seq_lookahead', [1, 2])
def test_context_creation(seq_lookahead):
    device = 'cuda'
    float_dtype = get_float(device)
    seq_length = 3
    seq_lookbehind = seq_length - seq_lookahead

    flock = prepare_flock_for_context(device=device, seq_length=seq_length, seq_lookahead=seq_lookahead,
                                      n_cluster_centers=3)

    nan = float("nan")

    # Randomise our important tensors
    flock.sp_flock.forward_clusters.uniform_()
    flock.tp_flock.passive_predicted_clusters_outputs.uniform_()
    flock.tp_flock.action_rewards.uniform_()
    flock.tp_flock.action_punishments.uniform_()

    flock._assemble_output_context()

    padding = torch.full((flock.params.flock_size, flock.n_cluster_centers),
                         fill_value=nan, dtype=float_dtype, device=device)

    expected_output_context = torch.cat(
        [
            flock.sp_flock.forward_clusters, flock.tp_flock.action_rewards, flock.tp_flock.action_punishments,
            flock.tp_flock.passive_predicted_clusters_outputs[:, seq_lookbehind, :], padding, padding
        ], dim=1).view(flock.params.flock_size, 2, NUMBER_OF_CONTEXT_TYPES, flock.n_cluster_centers)

    assert same(expected_output_context, flock.output_context)


@pytest.mark.parametrize('flock_size', [1, 3])
def test_run_flock_without_context_and_rewards(flock_size):
    device = 'cuda'
    float_dtype = get_float(device)
    seq_length = 4
    seq_lookahead = 2
    input_size = 7
    max_provider_context_size = 10
    max_provider_context_size2 = 1
    n_providers = 2
    n_cluster_centers = 3

    flock_with_context = prepare_flock_for_context(flock_size=flock_size, input_size=input_size, device=device,
                                                   seq_length=seq_length, seq_lookahead=seq_lookahead,
                                                   max_provider_context_size=max_provider_context_size,
                                                   n_providers=n_providers,
                                                   n_cluster_centers=n_cluster_centers)

    flock_without_context = prepare_flock_for_context(flock_size=flock_size, input_size=input_size, device=device,
                                                      seq_length=seq_length, seq_lookahead=seq_lookahead,
                                                      max_provider_context_size=max_provider_context_size2,
                                                      n_cluster_centers=n_cluster_centers,
                                                      n_providers=2)

    input_data = torch.rand((flock_size, input_size), device=device, dtype=float_dtype)
    input_context = torch.rand((flock_size, n_providers, NUMBER_OF_CONTEXT_TYPES,
                                max_provider_context_size), device=device, dtype=float_dtype)
    input_rewards = torch.rand((flock_size, 2), device=device, dtype=float_dtype)

    expected_rewards = torch.zeros((flock_size, 2), dtype=float_dtype, device=device)

    # without reward
    flock_with_context.run(input_data, input_context=input_context, input_rewards=None)
    flock_context = flock_with_context.tp_flock.input_context
    flock_rewards = flock_with_context.tp_flock.input_rewards

    assert same(expected_rewards, flock_rewards)
    assert same(input_context, flock_context)

    # without context
    flock_without_context.run(input_data, input_context=None, input_rewards=input_rewards)
    flock_context = flock_without_context.tp_flock.input_context
    flock_rewards = flock_without_context.tp_flock.input_rewards

    expected_flock_context = torch.zeros(flock_size, n_providers, NUMBER_OF_CONTEXT_TYPES, max_provider_context_size2,
                                         dtype=float_dtype, device=device)
    expected_flock_context[:, :, 0, :] = 1 / n_cluster_centers

    assert same(input_rewards, flock_rewards)
    assert same(expected_flock_context, flock_context)

    # without both
    flock_without_context.run(input_data, input_context=None, input_rewards=None)

    flock_context = flock_without_context.tp_flock.input_context
    flock_rewards = flock_without_context.tp_flock.input_rewards

    assert same(expected_flock_context, flock_context)
    assert same(expected_rewards, flock_rewards)


def test_run():
    """Test that the run method runs the reconstruction with correct indices."""
    device = 'cuda'
    float_dtype = get_float(device)
    flock_size = 3
    seq_length = 4
    seq_lookahead = 2
    input_size = 7
    max_provider_context_size = 1
    n_cluster_centers = 5
    compute_reconstruction = True

    flock = prepare_flock_for_context(flock_size=flock_size, input_size=input_size, device=device,
                                      seq_length=seq_length, seq_lookahead=seq_lookahead,
                                      max_provider_context_size=max_provider_context_size,
                                      n_cluster_centers=n_cluster_centers,
                                      compute_reconstruction=compute_reconstruction)

    input_data1 = torch.rand((flock_size, input_size), device=device, dtype=float_dtype)

    # replace the data for the second expert with new data, 1st expert will have the same data
    input_data2 = input_data1.clone()
    input_data2[1, :] = torch.rand((1, input_size), device=device, dtype=float_dtype)

    sp_outputs = torch.full((flock_size, n_cluster_centers), fill_value=-2, device=device, dtype=float_dtype)
    sp_reconstructions = torch.full((flock_size, input_size), fill_value=-2, device=device, dtype=float_dtype)

    # step 1 - all experts should reconstruct
    flock.sp_flock.forward_clusters.copy_(sp_outputs)
    flock.sp_flock.current_reconstructed_input.copy_(sp_reconstructions)
    flock.run(input_data1, input_context=None, input_rewards=None)
    for expert_id in range(flock_size):
        assert not same(sp_outputs[expert_id], flock.sp_flock.forward_clusters[expert_id])
        assert not same(sp_reconstructions[expert_id], flock.sp_flock.current_reconstructed_input[expert_id])

    # step 2 - just the expert no. 1 reconstruct
    flock.sp_flock.forward_clusters.copy_(sp_outputs)
    flock.sp_flock.current_reconstructed_input.copy_(sp_reconstructions)
    flock.run(input_data2, input_context=None, input_rewards=None)
    for expert_id in range(flock_size):
        if expert_id == 1:
            assert not same(sp_outputs[expert_id], flock.sp_flock.forward_clusters[expert_id])
            assert not same(sp_reconstructions[expert_id], flock.sp_flock.current_reconstructed_input[expert_id])
        else:
            assert same(sp_outputs[expert_id], flock.sp_flock.forward_clusters[expert_id])
            assert same(sp_reconstructions[expert_id], flock.sp_flock.current_reconstructed_input[expert_id])

    # step - none expert should reconstruct nor run, they have the same data
    flock.sp_flock.forward_clusters.copy_(sp_outputs)
    flock.sp_flock.current_reconstructed_input.copy_(sp_reconstructions)
    flock.run(input_data2, input_context=None, input_rewards=None)
    for expert_id in range(flock_size):
        assert same(sp_outputs[expert_id], flock.sp_flock.forward_clusters[expert_id])
        assert same(sp_reconstructions[expert_id], flock.sp_flock.current_reconstructed_input[expert_id])
