import pytest
import torch

from torchsim.core import get_float, SMALL_CONSTANT, FLOAT_NAN
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.expert_params import NUMBER_OF_CONTEXT_TYPES, EXPLORATION_REWARD
from torchsim.core.models.temporal_pooler import TPFlockForwardAndBackward, TPFlockBuffer
from torchsim.core.utils.tensor_utils import same, move_probs_towards_50_, normalize_probs_, \
    add_small_constant_

from torchsim.core.kernels import check_cuda_errors

eps = SMALL_CONSTANT


def create_tp_flock_forward_process(all_indices=None,
                                    frequent_seqs=None,
                                    frequent_seq_occurrences=None,
                                    frequent_seq_likelihoods_priors_clusters_context=None,
                                    flock_size=2,
                                    n_frequent_seqs=4,
                                    seq_length=3,
                                    seq_lookahead=1,
                                    n_cluster_centers=3,
                                    context_size=5,
                                    exploration_probability=0.00,
                                    cluster_exploration_prob=0.0,
                                    n_providers=1,
                                    own_rewards_weight=0.1,
                                    buffer=None,
                                    cluster_data=None,
                                    context_data=None,
                                    reward_data=None,
                                    projection_outputs=None,
                                    action_outputs=None,
                                    action_rewards=None,
                                    action_punishments=None,
                                    passive_predicted_cluster_outputs=None,
                                    frequent_context_likelihoods=None,
                                    frequent_rewards_punishments=None,
                                    frequent_explorations_attempts=None,
                                    frequent_explorations_results=None,
                                    execution_counter_forward=None,
                                    seq_likelihoods_by_context=None,
                                    best_matching_context=None,
                                    do_subflocking=True,
                                    produce_actions=False,
                                    follow_goals=False,
                                    compute_backward_pass=True,
                                    device='cuda'):
    float_dtype = get_float(device)
    if frequent_seqs is None:
        frequent_seqs = torch.zeros((flock_size, n_frequent_seqs, seq_length), dtype=float_dtype, device=device)

    if all_indices is None:
        all_indices = torch.arange(end=flock_size, dtype=torch.int64, device=device).unsqueeze(dim=1)

    if frequent_seq_occurrences is None:
        frequent_seq_occurrences = torch.zeros((flock_size, n_frequent_seqs), dtype=float_dtype, device=device)

    if frequent_seq_likelihoods_priors_clusters_context is None:
        frequent_seq_likelihoods_priors_clusters_context = torch.zeros((flock_size, n_frequent_seqs), dtype=float_dtype,
                                                                       device=device)

    if buffer is None:
        buffer = create_tp_buffer(flock_size=flock_size, device=device)

    if cluster_data is None:
        cluster_data = torch.zeros((flock_size, n_cluster_centers), device=device)

    if context_data is None:
        context_data = torch.zeros((flock_size, n_providers, NUMBER_OF_CONTEXT_TYPES, context_size), device=device)

    if reward_data is None:
        reward_data = torch.zeros((flock_size, 2), device=device)

    if projection_outputs is None:
        projection_outputs = torch.zeros((flock_size, n_cluster_centers), device=device)

    if action_outputs is None:
        action_outputs = torch.zeros((flock_size, n_cluster_centers), device=device)

    if action_rewards is None:
        action_rewards = torch.zeros((flock_size, n_cluster_centers), device=device)

    if action_punishments is None:
        action_punishments = torch.zeros((flock_size, n_cluster_centers), device=device)

    if passive_predicted_cluster_outputs is None:
        passive_predicted_cluster_outputs = torch.zeros((flock_size, seq_length, n_cluster_centers), device=device)

    if frequent_context_likelihoods is None:
        frequent_context_likelihoods = torch.zeros((flock_size, n_frequent_seqs, seq_length, n_providers, context_size),
                                                    dtype=float_dtype, device=device)

    if frequent_rewards_punishments is None:
        frequent_rewards_punishments = torch.zeros((flock_size, n_frequent_seqs, seq_lookahead, 2),
                                                    dtype=float_dtype, device=device)

    if frequent_explorations_attempts is None:
        frequent_explorations_attempts = torch.zeros((flock_size, n_frequent_seqs, seq_lookahead), dtype=float_dtype,
                                                     device=device)

    if frequent_explorations_results is None:
        frequent_explorations_results = torch.zeros((flock_size, n_frequent_seqs, seq_lookahead, n_cluster_centers),
                                                    dtype=float_dtype,
                                                    device=device)

    if execution_counter_forward is None:
        execution_counter_forward = torch.zeros((flock_size, 1), device=device, dtype=float_dtype)

    if seq_likelihoods_by_context is None:
        seq_likelihoods_by_context = torch.empty(flock_size, n_frequent_seqs, n_providers, context_size, dtype=float_dtype, device=device)
    if best_matching_context is None:
        best_matching_context = torch.empty((flock_size, n_providers, context_size), dtype=float_dtype, device=device)

    return TPFlockForwardAndBackward(all_indices,
                                     do_subflocking,
                                     buffer,
                                     cluster_data,
                                     context_data,
                                     reward_data,
                                     frequent_seqs,
                                     frequent_seq_occurrences,
                                     frequent_seq_likelihoods_priors_clusters_context,
                                     frequent_context_likelihoods,
                                     frequent_rewards_punishments,
                                     frequent_explorations_attempts,
                                     frequent_explorations_results,
                                     projection_outputs,
                                     action_outputs,
                                     action_rewards,
                                     action_punishments,
                                     passive_predicted_cluster_outputs,
                                     execution_counter_forward,
                                     seq_likelihoods_by_context,
                                     best_matching_context,
                                     n_frequent_seqs,
                                     n_cluster_centers,
                                     seq_length,
                                     seq_lookahead,
                                     context_size,
                                     n_providers,
                                     exploration_probability,
                                     own_rewards_weight,
                                     cluster_exploration_prob,
                                     device,
                                     produce_actions=produce_actions,
                                     follow_goals=follow_goals,
                                     compute_backward_pass=compute_backward_pass)


def create_tp_buffer(flock_size=2,
                     buffer_size=5,
                     n_cluster_centers=3,
                     n_frequent_seqs=4,
                     context_size=4,
                     n_providers=1,
                     device='cuda'):
    return TPFlockBuffer(AllocatingCreator(device),
                         flock_size,
                         buffer_size,
                         n_cluster_centers,
                         n_frequent_seqs,
                         context_size,
                         n_providers)


def test_forward_passive():
    device = 'cuda'
    float_dtype = get_float(device)
    flock_size = 2
    buffer_size = 6
    context_size = 2
    n_cluster_centers = 3
    seq_lookahead = 1
    n_frequent_seqs = 3
    n_providers = 1

    buffer = create_tp_buffer(flock_size=flock_size,
                              buffer_size=buffer_size,
                              n_cluster_centers=n_cluster_centers,
                              n_frequent_seqs=n_frequent_seqs,
                              context_size=context_size,
                              device=device)

    # region setup tensors

    nan = FLOAT_NAN
    invalid = -1
    small_const = SMALL_CONSTANT

    buffer.clusters.stored_data = torch.tensor([[[0, 1, 0],
                                                 [1, 0, 0],  # current_ptr
                                                 [0, 1, 0],
                                                 [1, 0, 0],
                                                 [0, 0, 1],
                                                 [1, 0, 0]],
                                                [[0, 1, 0],
                                                 [0, 0, 1],
                                                 [1, 0, 0],
                                                 [0, 0, 1],  # current_ptr
                                                 [nan, nan, nan],
                                                 [nan, nan, nan]]], dtype=float_dtype, device=device)
    normalize_probs_(buffer.clusters.stored_data, dim=2, add_constant=True)

    buffer.contexts.stored_data = torch.tensor([[[[1, 1]],
                                                 [[0.1, 0.1]],  # current_ptr
                                                 [[0, 1]],
                                                 [[1, 0.5]],
                                                 [[1, 0]],
                                                 [[0, 0]]],
                                                [[[1, 1]],
                                                 [[1, 1]],
                                                 [[0.9, 0]],
                                                 [[1, 0]],  # current_ptr
                                                 [[nan, nan]],
                                                 [[nan, nan]]]], dtype=float_dtype, device=device)
    move_probs_towards_50_(buffer.contexts.stored_data)

    # 2 is used just for checking that nothing else changed.
    buffer.seq_probs.stored_data.fill_(2)
    buffer.outputs.stored_data.fill_(2)

    buffer.current_ptr = torch.tensor([1, 3], dtype=torch.int64, device=device)
    buffer.total_data_written = torch.tensor([8, 3], dtype=torch.int64, device=device)

    frequent_seqs = torch.tensor([[[0, 1, 2],
                                   [1, 0, 1],
                                   [invalid, invalid, invalid]],
                                  [[0, 1, 0],
                                   [invalid, invalid, invalid],
                                   [invalid, invalid, invalid]]], dtype=torch.int64, device=device)

    frequent_seq_occurrences = torch.tensor([[5, 4, 0],
                                             [3, 0, 0]], dtype=float_dtype, device=device)

    frequent_context_likelihoods = torch.tensor([[[[[4.5, 2.5]], [[1, 2.5]], [[3, 2.5]]],
                                               [[[4, 2]], [[0.1, 2]], [[3, 2]]],
                                               [[[0, 0]], [[0, 0]], [[0, 0]]]],
                                              [[[[0, 1.5]], [[1, 1.5]], [[2.9, 1.5]]],
                                               [[[0, 0]], [[0, 0]], [[0, 0]]],
                                               [[[0, 0]], [[0, 0]], [[0, 0]]]]], dtype=float_dtype, device=device)
    add_small_constant_(frequent_context_likelihoods, small_const)

    cluster_data = torch.tensor([[0, 0, 1],
                                 [0.5, 0.5, 0]], dtype=float_dtype, device=device)

    # sequences:
    # [1, 0, 2],
    # [0, 2, 0 or 1]

    context_data = torch.tensor([[[[0, 1], [0, 0], [0, 0]]],
                                 [[[1, 0.5], [0, 0], [0, 0]]]], dtype=float_dtype, device=device)

    # contexts:
    # [[1.0, 1.0], [0.1, 0.1], [0.0, 1.0]],
    # [[0.9, 0.0], [1.0, 0.0], [1.0, 0.5]]

    frequent_exploration_results = torch.full((flock_size, n_frequent_seqs, seq_lookahead, n_cluster_centers),
                                              fill_value=0.5,
                                              dtype=float_dtype, device=device)

    # Pre-fill the output tensors so that we can check that they were written into
    projection_output = torch.full((flock_size, n_cluster_centers), fill_value=-2, dtype=float_dtype, device=device)
    action_output = torch.full((flock_size, n_cluster_centers), fill_value=-2, dtype=float_dtype,
                               device=device)

    # endregion

    process = create_tp_flock_forward_process(frequent_seqs=frequent_seqs,
                                              frequent_seq_occurrences=frequent_seq_occurrences,
                                              flock_size=flock_size,
                                              n_frequent_seqs=n_frequent_seqs,
                                              seq_length=3,
                                              seq_lookahead=seq_lookahead,
                                              n_cluster_centers=n_cluster_centers,
                                              context_size=context_size,
                                              exploration_probability=0,
                                              buffer=buffer,
                                              cluster_data=cluster_data,
                                              context_data=context_data,
                                              projection_outputs=projection_output,
                                              action_outputs=action_output,
                                              frequent_context_likelihoods=frequent_context_likelihoods,
                                              frequent_explorations_results=frequent_exploration_results,
                                              do_subflocking=True,
                                              n_providers=n_providers,
                                              device=device)

    process.run_and_integrate()

    # temporary process expected values

    # Flock 1: The first sequence differs in one cluster, the second in two clusters, the third is invalid.
    expected_seq_likelihoods_clusters_only = torch.tensor([[eps, eps ** 2, 0],
                                                           [0.5 * eps, 0, 0]], dtype=float_dtype, device=device)

    expected_seq_likelihoods_without_context = torch.tensor([[5 * eps, 4 * (eps ** 2), 0],
                                                             [3 * 0.5 * eps, 0, 0]], dtype=float_dtype, device=device)

    # region expected_values
    expected_buffer_current_ptr = torch.tensor([2, 4], dtype=torch.int64, device=device)
    expected_buffer_total_data_written = torch.tensor([9, 4], dtype=torch.int64, device=device)

    expected_buffer_clusters = torch.tensor([[[0, 1, 0],
                                              [1, 0, 0],
                                              [0, 0, 1],  # current_ptr
                                              [1, 0, 0],
                                              [0, 0, 1],
                                              [1, 0, 0]],
                                             [[0, 1, 0],
                                              [0, 0, 1],
                                              [1, 0, 0],
                                              [0, 0, 1],
                                              [0.5, 0.5, 0],  # current_ptr
                                              [nan, nan, nan]]], dtype=float_dtype, device=device)
    normalize_probs_(expected_buffer_clusters, dim=2, add_constant=True)

    expected_buffer_contexts = torch.tensor([[[[1, 1]],
                                              [[0.1, 0.1]],
                                              [[0, 1]],  # current_ptr
                                              [[1, 0.5]],
                                              [[1, 0]],
                                              [[0, 0]]],
                                             [[[1, 1]],
                                              [[1, 1]],
                                              [[0.9, 0]],
                                              [[1, 0]],
                                              [[1, 0.5]],  # current_ptr
                                              [[nan, nan]]]], dtype=float_dtype, device=device)
    move_probs_towards_50_(expected_buffer_contexts)

    # There are 3 frequent sequences, so current pointer says that the first one is 100% probable.
    expected_buffer_seq_probs = torch.tensor([[[2, 2, 2],
                                               [2, 2, 2],
                                               [0.9999, 0.0001, 0],  # current_ptr
                                               [2, 2, 2],
                                               [2, 2, 2],
                                               [2, 2, 2]],
                                              [[2, 2, 2],
                                               [2, 2, 2],
                                               [2, 2, 2],
                                               [2, 2, 2],
                                               [1, 0, 0],  # current_ptr
                                               [2, 2, 2]]], dtype=float_dtype, device=device)

    # There are 3 cluster centers.
    expected_buffer_outputs = torch.tensor([[[2, 2, 2],
                                             [2, 2, 2],
                                             [0.25, 0.5, 0.25],  # current_ptr
                                             [2, 2, 2],
                                             [2, 2, 2],
                                             [2, 2, 2]],
                                            [[2, 2, 2],
                                             [2, 2, 2],
                                             [2, 2, 2],
                                             [2, 2, 2],
                                             [0.5, 0.5, 0],  # current_ptr
                                             [2, 2, 2]]], dtype=float_dtype, device=device)

    expected_projection_output = torch.tensor([[0.25, 0.5, 0.25],
                                               [0.5, 0.5, 0]], dtype=float_dtype, device=device)

    expected_action_output = torch.tensor([[0, 0, 1],
                                           [1, 0, 0]], dtype=float_dtype, device=device)

    # endregion

    assert same(expected_seq_likelihoods_clusters_only, process.seq_likelihoods_clusters, eps=1e-3)
    assert same(expected_seq_likelihoods_without_context, process.seq_likelihoods_priors_clusters, eps=1e-3)

    assert same(expected_projection_output, projection_output, eps=1e-3)
    assert same(expected_action_output, action_output, eps=1e-3)
    # test also storing into buffer
    assert same(expected_buffer_current_ptr, buffer.current_ptr)
    assert same(expected_buffer_total_data_written, buffer.total_data_written)
    assert same(expected_buffer_outputs, buffer.outputs.stored_data, eps=1e-3)
    assert same(expected_buffer_seq_probs, buffer.seq_probs.stored_data, eps=1e-3)
    assert same(expected_buffer_clusters, buffer.clusters.stored_data, eps=1e-3)
    assert same(expected_buffer_contexts, buffer.contexts.stored_data, eps=1e-3)


def setup_forward_passive_vs_active(follow_goals, device='cuda'):
    float_dtype = get_float(device)
    flock_size = 1
    buffer_size = 6
    context_size = 2
    n_cluster_centers = 3
    seq_lookahead = 1
    n_frequent_seqs = 3
    seq_length = 3
    n_providers = 1

    buffer = create_tp_buffer(flock_size=flock_size,
                              buffer_size=buffer_size,
                              n_cluster_centers=n_cluster_centers,
                              n_frequent_seqs=n_frequent_seqs,
                              context_size=context_size,
                              device=device,
                              n_providers=n_providers)

    # region setup tensors
    invalid = -1
    small_const = SMALL_CONSTANT

    buffer.clusters.stored_data = torch.tensor([[[0, 1, 0],
                                                 [1, 0, 0],
                                                 [0, 1, 0],
                                                 [0, 0, 1],
                                                 [0, 1, 0],
                                                 [1, 0, 0]]], dtype=float_dtype, device=device)  # current_ptr
    normalize_probs_(buffer.clusters.stored_data, dim=2, add_constant=True)

    buffer.contexts.stored_data = torch.tensor([[[[0, 0]],
                                                 [[0.1, 0.1]],
                                                 [[0, 1]],
                                                 [[0, 0.5]],
                                                 [[0.5, 0.5]],
                                                 [[1, 0]]]], dtype=float_dtype, device=device)  # current_ptr
    move_probs_towards_50_(buffer.contexts.stored_data)

    # 2 is used just for checking that nothing else changed.
    buffer.seq_probs.stored_data.fill_(2)
    buffer.outputs.stored_data.fill_(2)

    buffer.current_ptr = torch.tensor([5], dtype=torch.int64, device=device)
    buffer.total_data_written = torch.tensor([8], dtype=torch.int64, device=device)

    # Passive model should say that seq 1 is the most likely one
    frequent_seqs = torch.tensor([[[0, 1, 2],
                                   [0, 1, 0],
                                   [invalid, invalid, invalid]]], dtype=torch.int64, device=device)

    frequent_seq_occurrences = torch.tensor([[5, 4, 0]], dtype=float_dtype, device=device)

    # Pos contexts: 0= [1,0], 1= [0,0], 2= [0,1]
    frequent_context_likelihoods = torch.tensor([[[[[1, 0.01]], [[0.01, 1]], [[0.01, 5]]],
                                               [[[0.5, 0.5]], [[0.01, 1]], [[4, 0.01]]],
                                               [[[0., 0.]], [[0., 0.]], [[0., 0.]]]]], dtype=float_dtype, device=device)
    add_small_constant_(frequent_context_likelihoods, small_const)

    # Cluster 1 is the input
    cluster_data = torch.tensor([[0, 1, 0]], dtype=float_dtype, device=device)

    # This goal context says: "Go to cluster 0 next"
    context_data = torch.tensor([[[[0, 1], [5, 0], [0, 0]]]], dtype=float_dtype, device=device)

    frequent_exploration_results = torch.tensor([[[[0, 0, 1]],
                                                  [[1, 0, 0]],
                                                  [[0, 0, 0]]]], dtype=float_dtype, device=device)

    frequent_exploration_results += 1 / n_cluster_centers


    # Pre-fill the output tensors so that we can check that they were written into
    projection_output = torch.full((flock_size, n_cluster_centers), fill_value=-2, dtype=float_dtype, device=device)
    action_output = torch.full((flock_size, n_cluster_centers), fill_value=-2, dtype=float_dtype,
                               device=device)
    passive_predicted_cluster_outputs = torch.full((flock_size, seq_length, n_cluster_centers),
                                                   fill_value=2, dtype=float_dtype, device=device)

    # endregion

    process = create_tp_flock_forward_process(frequent_seqs=frequent_seqs,
                                              frequent_seq_occurrences=frequent_seq_occurrences,
                                              flock_size=flock_size,
                                              n_frequent_seqs=n_frequent_seqs,
                                              seq_length=seq_length,
                                              seq_lookahead=seq_lookahead,
                                              n_cluster_centers=n_cluster_centers,
                                              context_size=context_size,
                                              exploration_probability=0,
                                              buffer=buffer,
                                              cluster_data=cluster_data,
                                              context_data=context_data,
                                              projection_outputs=projection_output,
                                              action_outputs=action_output,
                                              frequent_context_likelihoods=frequent_context_likelihoods,
                                              frequent_explorations_results=frequent_exploration_results,
                                              passive_predicted_cluster_outputs=passive_predicted_cluster_outputs,
                                              do_subflocking=True,
                                              follow_goals=follow_goals,
                                              produce_actions=True,
                                              device=device)

    return buffer, process, projection_output, action_output, passive_predicted_cluster_outputs


def test_forward_passive_vs_active():
    """This test contrasts the passive model with the active one (controlled by the goal context).

    The expert has somehow learned two sequences [0, 1, 2], and [0, 1, 0] and receives the sequence [0, 1].
    The context for  cluster 0 in sequence [0, 1, 2] is always a sharp [1, 0], but for [0, 1, 0], varies between
    [1, 0] and [0, 1]. In this case, the model has seen [0, [1, 0]] and [1, [0, 1]], this means that passively,
    sequence [0, 1, 2] is more likely. And the action (as this is bottom level) is to try and follow the sequence
    ([0, 0, 1]).

    The active model has a reward associated with sequence [0, 1, 0] however. So when goal directed behaviour is active,
    the received context from the parent indicates that the child should go for that reward instead of following the
    passive model.

    In this test, the passive model sees the cluster 1, and knows that the previous cluster was 0, so left to the
    passive model, it predicts that the next cluster/action will be 2.

    When running on only goal directed behaviour the active model will see that, like the passive model, the current
    cluster is 1 and the previous is 0. But it also sees that reward will be due if it can go to cluster 0 again, so in
    this case the action is to move to cluster 0.
    """
    device = 'cuda'
    float_dtype = get_float(device)

    # Test the passive model first
    buffer, process, projection_output, action_output, passive_prediction_outputs = setup_forward_passive_vs_active(
        follow_goals=False, device=device)

    process.run_and_integrate()

    # region passive expected_values
    expected_seq_likelihoods_clusters_only = torch.tensor([[0.9996, 0.9996, 0]], dtype=float_dtype, device=device)

    expected_seq_likelihoods_without_context = torch.tensor([[4.9980, 3.9984, 0.0000]], dtype=float_dtype,
                                                            device=device)

    expected_buffer_current_ptr = torch.tensor([0], dtype=torch.int64, device=device)
    expected_buffer_total_data_written = torch.tensor([9], dtype=torch.int64, device=device)

    expected_buffer_clusters = torch.tensor([[[0, 1, 0],
                                              [1, 0, 0],
                                              [0, 1, 0],
                                              [0, 0, 1],
                                              [0, 1, 0],
                                              [1, 0, 0]]], dtype=float_dtype, device=device)
    normalize_probs_(expected_buffer_clusters, dim=2, add_constant=True)

    expected_buffer_contexts = torch.tensor([[[[0, 1]],
                                              [[0.1, 0.1]],
                                              [[0, 1]],
                                              [[0, 0.5]],
                                              [[0.5, 0.5]],
                                              [[1, 0]]]], dtype=float_dtype, device=device)
    move_probs_towards_50_(expected_buffer_contexts)

    # There are 3 frequent sequences, so current pointer says that the first one is 75% probable.
    expected_buffer_seq_probs = torch.tensor([[[0.7142, 0.2858, 0],  # current_ptr
                                               [2, 2, 2],
                                               [2, 2, 2],
                                               [2, 2, 2],
                                               [2, 2, 2],
                                               [2, 2, 2]]], dtype=float_dtype, device=device)

    # There are 3 cluster centers.
    expected_buffer_outputs = torch.tensor([[[0.3214, 0.5, 0.1786],  # current_ptr
                                             [2, 2, 2],
                                             [2, 2, 2],
                                             [2, 2, 2],
                                             [2, 2, 2],
                                             [2, 2, 2]]], dtype=float_dtype, device=device)

    expected_passive_predicted_clusters_outputs = torch.tensor([[1, 0, 0], [0, 1, 0], [0.2858, 0, 0.7142]],
                                                               dtype=float_dtype,
                                                               device=device).unsqueeze(0)  # add the flock_size=1

    expected_projection_output = torch.tensor([[0.3214, 0.5, 0.1786]], dtype=float_dtype, device=device)

    expected_action_output = torch.tensor([[0.2858, 0, 0.7142]], dtype=float_dtype, device=device)

    # endregion

    # region passive tests

    assert same(expected_seq_likelihoods_clusters_only, process.seq_likelihoods_clusters, eps=1e-1)
    assert same(expected_seq_likelihoods_without_context, process.seq_likelihoods_priors_clusters, eps=1e-1)
    assert same(expected_passive_predicted_clusters_outputs, passive_prediction_outputs, eps=1e-3)

    assert same(expected_projection_output, projection_output, eps=1e-3)
    # This is the main difference between passive and active
    assert same(expected_action_output, action_output, eps=1e-3)
    # test also storing into buffer
    assert same(expected_buffer_current_ptr, buffer.current_ptr)
    assert same(expected_buffer_total_data_written, buffer.total_data_written)
    assert same(expected_buffer_outputs, buffer.outputs.stored_data, eps=1e-3)
    assert same(expected_buffer_seq_probs, buffer.seq_probs.stored_data, eps=1e-3)
    assert same(expected_buffer_clusters, buffer.clusters.stored_data, eps=1e-3)
    assert same(expected_buffer_contexts, buffer.contexts.stored_data, eps=1e-3)

    # endregion

    # Test the active model
    buffer, process, projection_output, action_output, passive_prediction_outputs = setup_forward_passive_vs_active(
        follow_goals=True, device=device)

    process.run_and_integrate()

    # region active expected_values
    expected_seq_likelihoods_clusters_only = torch.tensor([[0.9996, 0.9996, 0]], dtype=float_dtype, device=device)

    expected_seq_likelihoods_without_context = torch.tensor([[4.9980, 3.9984, 0.0000]], dtype=float_dtype,
                                                            device=device)

    expected_buffer_current_ptr = torch.tensor([0], dtype=torch.int64, device=device)
    expected_buffer_total_data_written = torch.tensor([9], dtype=torch.int64, device=device)

    expected_buffer_clusters = torch.tensor([[[0, 1, 0],
                                              [1, 0, 0],
                                              [0, 1, 0],
                                              [0, 0, 1],
                                              [0, 1, 0],
                                              [1, 0, 0]]], dtype=float_dtype, device=device)
    normalize_probs_(expected_buffer_clusters, dim=2, add_constant=True)

    expected_buffer_contexts = torch.tensor([[[[0, 1]],
                                              [[0.1, 0.1]],
                                              [[0, 1]],
                                              [[0, 0.5]],
                                              [[0.5, 0.5]],
                                              [[1, 0]]]], dtype=float_dtype, device=device)
    move_probs_towards_50_(expected_buffer_contexts)

    # There are 3 frequent sequences, so current pointer says that the first one is 75% probable.
    expected_buffer_seq_probs = torch.tensor([[[0.7142, 0.2858, 0],  # current_ptr
                                               [2, 2, 2],
                                               [2, 2, 2],
                                               [2, 2, 2],
                                               [2, 2, 2],
                                               [2, 2, 2]]], dtype=float_dtype, device=device)

    # There are 3 cluster centers.
    expected_buffer_outputs = torch.tensor([[[0.3214, 0.5, 0.1786],  # current_ptr
                                             [2, 2, 2],
                                             [2, 2, 2],
                                             [2, 2, 2],
                                             [2, 2, 2],
                                             [2, 2, 2]]], dtype=float_dtype, device=device)

    expected_projection_output = torch.tensor([[0.3214, 0.5, 0.1786]], dtype=float_dtype, device=device)

    expected_action_output = torch.tensor([[0.6133, 0., 0.3867]], dtype=float_dtype, device=device)

    # endregion

    # region active tests
    assert same(expected_seq_likelihoods_clusters_only, process.seq_likelihoods_clusters, eps=1e-2)
    assert same(expected_seq_likelihoods_without_context, process.seq_likelihoods_priors_clusters, eps=1e-1)
    assert same(expected_passive_predicted_clusters_outputs, passive_prediction_outputs, eps=1e-3)

    assert same(expected_projection_output, projection_output, eps=1e-3)
    # This is the main difference between passive and active
    assert same(expected_action_output, action_output, eps=1e-2)

    # test also storing into buffer
    assert same(expected_buffer_current_ptr, buffer.current_ptr)
    assert same(expected_buffer_total_data_written, buffer.total_data_written)
    assert same(expected_buffer_outputs, buffer.outputs.stored_data, eps=1e-3)
    assert same(expected_buffer_seq_probs, buffer.seq_probs.stored_data, eps=1e-3)
    assert same(expected_buffer_clusters, buffer.clusters.stored_data, eps=1e-3)
    assert same(expected_buffer_contexts, buffer.contexts.stored_data, eps=1e-3)

    # endregion


# region compute_seq_likelihoods

def test_compute_seq_likelihoods():
    flock_size = 2
    n_frequent_seqs = 4
    n_cluster_centers = 3
    seq_length = 3
    seq_lookahead = 1
    seq_lookbehind = seq_length - seq_lookahead
    device = 'cuda'
    float_dtype = get_float(device)
    do_subflocking = True
    context_size = 2
    n_providers = 2

    # region Setup

    invalid = -1

    buffer = create_tp_buffer(flock_size=flock_size,
                              context_size=context_size,
                              device=device,
                              n_providers=n_providers)

    buffer.current_ptr = torch.tensor([0, 4], dtype=torch.int64, device=device)
    buffer.clusters.stored_data = torch.tensor([[[0, 0.1, 0.9],
                                                 [1, 0, 0],
                                                 [0, 1, 0],
                                                 [1, 0, 0],
                                                 [0, 0.5, 0.5]],
                                                [[0, 1, 0],
                                                 [0, 0, 1],
                                                 [1, 0, 0],
                                                 [0.2, 0.4, 0.4],
                                                 [0.5, 0, 0.5]]], dtype=float_dtype, device=device)
    normalize_probs_(buffer.clusters.stored_data, dim=2, add_constant=True)

    buffer.contexts.stored_data = torch.tensor([[[[0.2, 0.1], [0.3, 0.4]],
                                                 [[1, 0], [1, 0.2]],
                                                 [[0, 1], [0.3, 0.4]],
                                                 [[0, 1], [1, 0.2]],
                                                 [[0, 1], [0.5, 0.5]]],
                                                [[[1, 0.3], [1, 0.5]],
                                                 [[0.2, 0.4], [0, 1]],
                                                 [[0, 1], [1, 0.6]],
                                                 [[0.5, 0.5], [0.4, 0.2]],
                                                 [[0.2, 0.4], [0.1, 0.9]]]], dtype=float_dtype, device=device)
    move_probs_towards_50_(buffer.contexts.stored_data)

    frequent_seqs = torch.tensor([[[1, 0, 1],
                                   [0, 1, 2],
                                   [1, 2, 0],
                                   [invalid, invalid, invalid]],
                                  [[1, 0, 1],
                                   [0, 1, 2],
                                   [1, 2, 0],
                                   [0, 2, 1]]], dtype=torch.int64, device=device)

    frequent_seq_occurrences = torch.tensor([[3, 2, 1, 0],
                                             [10, 2, 2, 0.1]], dtype=float_dtype, device=device)

    frequent_context_likelihoods = torch.tensor([[[[[0.5, 0.5], [0.5, 0.5]],
                                                [[0.5, 0.5], [0.5, 0.5]],
                                                [[0.5, 0.5], [0.5, 0.5]]],

                                               [[[0.5238, 0.6], [0.375, 0.2]],
                                                [[0.3023, 0.2], [0.2593, 0.4]],
                                                [[0.2308, 0.5], [0.2105, 1]]],

                                               [[[0.1, 1], [0.1, 1]],
                                                [[0.1, 1], [0.1, 1]],
                                                [[0.1, 1], [0.1, 1]]],

                                               [[[0, 0], [0, 0]],
                                                [[0, 0], [0, 0]],
                                                [[0, 0], [0, 0]]]],

                                              [[[[0.2, 0.5], [0.2, 0.4]],
                                                [[0.4, 0], [0.1, 0.1]],
                                                [[1, 0], [0.9, 0.3]]],

                                               [[[0.2, 0.1], [0.3333, 0.3]],
                                                [[0.1, 0.9], [0.1, 0.4]],
                                                [[0.1, 0.4], [0.1, 0.7]]],

                                               [[[0.1, 0.4], [0.1, 1]],
                                                [[0.4, 0.1], [0.1, 1]],
                                                [[0.1, 0.6], [0.1, 1]]],

                                               [[[0.1, 0.54], [0.1, 0.67]],
                                                [[0.1, 0.31], [0.1, 0.62]],
                                                [[0.1, 0.62], [33333, 0.552]]]]], dtype=float_dtype, device=device)
    add_small_constant_(frequent_context_likelihoods, 1e-4)

    seq_likelihoods_clusters = torch.full((flock_size, n_frequent_seqs),
                                          fill_value=-1, dtype=float_dtype, device=device)

    seq_likelihoods_priors_clusters = torch.full((flock_size, n_frequent_seqs),
                                                 fill_value=-1, dtype=float_dtype, device=device)

    seq_likelihoods_for_each_provider = torch.zeros((flock_size, n_frequent_seqs, seq_lookbehind, n_providers),
                                                    dtype=float_dtype, device=device)

    seq_likelihoods_priors_clusters_context = torch.full((flock_size, n_frequent_seqs),
                                                         fill_value=-1, dtype=float_dtype, device=device)

    seq_probs_clusters_context = torch.full((flock_size, n_frequent_seqs),
                                            fill_value=-1, dtype=float_dtype, device=device)

    all_indices = torch.arange(end=2, dtype=torch.int64, device=device).unsqueeze(dim=1)

    cluster_data = torch.zeros((flock_size, n_cluster_centers), dtype=float_dtype, device=device)
    context_data = torch.zeros((flock_size, n_providers, NUMBER_OF_CONTEXT_TYPES, context_size), dtype=float_dtype,
                               device=device)
    projection_outputs = torch.zeros((flock_size, n_cluster_centers), dtype=float_dtype, device=device)
    action_outputs = torch.zeros((flock_size, n_cluster_centers), dtype=float_dtype, device=device)

    provider_informativeness = torch.full((flock_size, seq_lookbehind, n_providers),
                                          fill_value=-1, dtype=float_dtype, device=device)

    process = create_tp_flock_forward_process(all_indices=all_indices,
                                              do_subflocking=do_subflocking,
                                              buffer=buffer,
                                              cluster_data=cluster_data,
                                              context_data=context_data,
                                              frequent_seqs=frequent_seqs,
                                              frequent_seq_occurrences=frequent_seq_occurrences,
                                              frequent_context_likelihoods=frequent_context_likelihoods,
                                              projection_outputs=projection_outputs,
                                              action_outputs=action_outputs,
                                              n_frequent_seqs=n_frequent_seqs,
                                              n_cluster_centers=n_cluster_centers,
                                              seq_length=seq_length,
                                              seq_lookahead=seq_lookahead,
                                              context_size=context_size,
                                              n_providers=n_providers,
                                              device=device)

    # endregion

    # region Expected values

    expected_cluster_history = torch.tensor([[[0, 0.5, 0.5],
                                              [0, 0.1, 0.9]],
                                             [[0.2, 0.4, 0.4],
                                              [0.5, 0, 0.5]]], dtype=float_dtype, device=device)
    normalize_probs_(expected_cluster_history, dim=2, add_constant=True)

    expected_context_history = torch.tensor([[[[0, 1], [0.5, 0.5]],
                                              [[0.2, 0.1], [0.3, 0.4]]],
                                             [[[0.5, 0.5], [0.4, 0.2]],
                                              [[0.2, 0.4], [0.1, 0.9]]]], dtype=float_dtype, device=device)
    move_probs_towards_50_(expected_context_history)

    expected_seq_likelihoods_clusters = torch.tensor([[0, 0, 0.45, 0],
                                                      [0.2, 0, 0.2, 0.1]],
                                                     dtype=float_dtype, device=device)

    expected_seq_likelihoods_for_each_provider = torch.tensor([[[[0, 0], [0, 0]],
                                                                [[0, 0], [0, 0]],
                                                                [[0.44, 0.24], [0.054, 0.19]],
                                                                [[0, 0], [0, 0]]],
                                                               [[[0.7, 0.32], [0.16, 0.2]],
                                                                [[0, 0], [0, 0]],
                                                                [[0.1, 0.096], [0.048, 0.36]],
                                                                [[0, 0], [0, 0]]]], dtype=float_dtype, device=device)

    expected_likelihood_priors_clusters = torch.tensor([[0, 0, 0.45, 0],
                                                        [2, 0, 0.4, 0.01]], dtype=float_dtype, device=device)

    expected_provider_informativeness = torch.tensor([[[0., 0.], [0., 0.]],
                                                      [[0.007, 0.012], [0.013, 0.48]]], dtype=float_dtype,
                                                     device=device)

    expected_seq_likelihoods_priors_clusters_context = torch.tensor([[0, 0, 0.44, 0],
                                                                     [0.2, 0, 0.36, 0.0]], dtype=float_dtype,
                                                                    device=device)

    expected_seq_probs_clusters_context = torch.tensor([[0, 0, 1, 0],
                                                        [0.077, 0, 0.7, 0.21]], dtype=float_dtype, device=device)

    seq_likelihoods_by_context = torch.zeros(flock_size, n_frequent_seqs, n_providers, context_size, dtype=float_dtype,
                                             device=device)
    # endregion

    process.cluster_history.fill_(-1)
    process.context_history.fill_(-1)

    process._compute_seq_likelihoods(buffer,
                                     process.cluster_history,
                                     process.context_history,
                                     seq_likelihoods_clusters,
                                     seq_likelihoods_priors_clusters,
                                     seq_likelihoods_for_each_provider,
                                     seq_likelihoods_by_context,
                                     seq_likelihoods_priors_clusters_context,
                                     seq_probs_clusters_context,
                                     frequent_seqs,
                                     frequent_seq_occurrences,
                                     frequent_context_likelihoods,
                                     provider_informativeness)

    check_cuda_errors()

    assert same(expected_context_history, process.context_history, eps=1e-1)
    assert same(expected_seq_likelihoods_clusters, seq_likelihoods_clusters, eps=1e-1)
    assert same(expected_likelihood_priors_clusters, seq_likelihoods_priors_clusters, eps=1e-2)
    assert same(expected_seq_likelihoods_for_each_provider, seq_likelihoods_for_each_provider, eps=1e-2)
    assert same(expected_seq_likelihoods_priors_clusters_context, seq_likelihoods_priors_clusters_context, eps=1e-1)
    assert same(expected_provider_informativeness, provider_informativeness, eps=1e-2)
    assert same(expected_seq_probs_clusters_context, seq_probs_clusters_context, eps=1e-2)


def test_compute_seq_likelihoods_priors_clusters():
    device = 'cuda'
    float_dtype = get_float(device)
    flock_size = 2
    n_cluster_centers = 4
    n_frequent_seqs = 3
    seq_lookahead = 1
    seq_length = 3
    seq_lookbehind = seq_length - seq_lookahead
    buffer_size = 3

    buffer = create_tp_buffer(flock_size=flock_size, buffer_size=buffer_size, n_cluster_centers=n_cluster_centers,
                              n_frequent_seqs=n_frequent_seqs, device=device)

    buffer.clusters.stored_data = torch.tensor([[[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]],
                                                [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]], dtype=float_dtype,
                                               device=device)

    process = create_tp_flock_forward_process(flock_size=flock_size, n_cluster_centers=n_cluster_centers,
                                              n_frequent_seqs=n_frequent_seqs, seq_lookahead=seq_lookahead,
                                              seq_length=seq_length, buffer=buffer, device=device)

    cluster_history = torch.zeros((flock_size, seq_lookbehind, n_cluster_centers), dtype=float_dtype, device=device)
    seq_likelihoods_clusters = torch.zeros((flock_size, n_frequent_seqs), dtype=float_dtype, device=device)
    seq_likelihoods_priors_clusters = torch.zeros((flock_size, n_frequent_seqs), dtype=float_dtype, device=device)

    frequent_seqs = torch.tensor([[[1, 0, 2], [0, 2, 1], [-1, -1, -1]],
                                  [[2, 1, 0], [0, 2, 1], [1, 0, 2]]], dtype=torch.int64, device=device)
    frequent_seq_occurrences = torch.tensor([[2, 3, 0], [4, 1, 1]], dtype=float_dtype, device=device)

    process._compute_seq_likelihoods_priors_clusters(buffer, cluster_history, seq_likelihoods_clusters,
                                                     seq_likelihoods_priors_clusters, frequent_seqs,
                                                     frequent_seq_occurrences)

    expected_cluster_history = torch.tensor([[[1, 0, 0, 0], [0, 0, 1, 0]],
                                             [[0, 1, 0, 0], [1, 0, 0, 0]]], dtype=float_dtype, device=device)
    expected_seq_likelihoods_clusters = torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=float_dtype, device=device)
    expected_seq_likelihoods_priors_clusters = torch.tensor([[0, 3, 0, ], [0, 0, 1]], dtype=float_dtype, device=device)

    assert same(expected_cluster_history, cluster_history)
    assert same(expected_seq_likelihoods_clusters, seq_likelihoods_clusters)
    assert same(expected_seq_likelihoods_priors_clusters, seq_likelihoods_priors_clusters)


def test_compute_seq_likelihoods_for_each_provider():
    device = 'cuda'
    float_dtype = get_float(device)
    flock_size = 2
    n_cluster_centers = 4
    n_frequent_seqs = 3
    seq_lookahead = 1
    seq_length = 3
    seq_lookbehind = seq_length - seq_lookahead
    buffer_size = 3
    context_size = 3
    n_providers = 2

    buffer = create_tp_buffer(flock_size=flock_size, buffer_size=buffer_size, n_cluster_centers=n_cluster_centers,
                              n_frequent_seqs=n_frequent_seqs, context_size=context_size, device=device,
                              n_providers=n_providers)

    buffer.contexts.stored_data = torch.tensor([[[[1, 0, 0], [1, 0, 0]],
                                                 [[1, 0, 0], [0, 0, 1]],
                                                 [[0, 1, 0], [0, 1, 0]]],

                                                [[[1, 0, 0], [1, 0, 0]],
                                                 [[1, 0, 0], [0, 1, 0]],
                                                 [[0, 1, 0], [0, 0, 1]]]], dtype=float_dtype, device=device)

    process = create_tp_flock_forward_process(flock_size=flock_size, n_cluster_centers=n_cluster_centers,
                                              n_frequent_seqs=n_frequent_seqs, seq_lookahead=seq_lookahead,
                                              seq_length=seq_length, context_size=context_size, buffer=buffer,
                                              n_providers=n_providers,
                                              device=device)

    context_history = torch.zeros((flock_size, seq_lookbehind, n_providers, context_size),
                                  dtype=float_dtype, device=device)

    seq_likelihoods_priors_clusters = torch.tensor([[3, 3, 0], [2, 1, 1]], dtype=float_dtype, device=device)

    # Largest likelihoods before priors:
    # Provider 0: F0: [S1], F1: [S1, S2]
    # Provider 1: F0: [S0], F1: [S0],
    frequent_context_likelihoods = torch.tensor([[[[[0.0, 0.1, 0.4], [0, 0, 1]],
                                                [[0.5, 0.5, 0], [1, 0, 0]],
                                                [[0.7, 0.3, 0], [0, 1, 0]]],

                                               [[[1, 0, 0], [0, 1, 0]],
                                                [[0.4, 0.6, 0], [0, 1, 0]],
                                                [[0.1, 0.9, 0], [0, 1, 0]]],

                                               [[[0, 0, 0], [0, 0, 0]],
                                                [[0, 0, 0], [0, 0, 0]],
                                                [[0, 0, 0], [0, 0, 0]]]],

                                              [[[[0.4, 0.6, 0], [0, 1, 0]],
                                                [[0.2, 0.4, 0.4], [0, 1, 0]],
                                                [[0.4, 0.3, 0.3], [1, 0, 0]]],

                                               [[[0.7, 0.2, 0.1], [0.8, 0.2, 0]],
                                                [[1, 0., 0], [1, 0, 0]],
                                                [[0.2, 0.6, 0], [1, 0, 0]]],

                                               [[[1, 0., 0], [0, 0, 1]],
                                                [[0.7, 0.3, 0], [0, 0, 1]],
                                                [[0.2, 0.8, 0], [0, 0, 1]]]]], dtype=float_dtype,
                                             device=device)

    seq_likelihoods_for_each_provider = torch.zeros((flock_size, n_frequent_seqs, seq_lookbehind, n_providers),
                                                    dtype=float_dtype, device=device)

    seq_likelihoods_by_context = torch.zeros(flock_size, n_frequent_seqs, n_providers, context_size, dtype=float_dtype,
                                             device=device)

    process._compute_seq_likelihoods_for_each_provider(buffer, context_history, seq_likelihoods_priors_clusters,
                                                       frequent_context_likelihoods,
                                                       seq_likelihoods_for_each_provider, seq_likelihoods_by_context)

    expected_seq_likelihoods_for_each_provider = torch.tensor([[[[0., 3],
                                                                 [1.5, 0.]],
                                                                [[3, 0.],
                                                                 [1.8, 3]],
                                                                [[0., 0.],
                                                                 [0., 0.]]],
                                                               [[[0.8, 2.],
                                                                 [0.8, 0.]],
                                                                [[0.7, 0.2],
                                                                 [0., 0.]],
                                                                [[1., 0.],
                                                                 [0.3, 1.]]]], dtype=float_dtype,
                                                              device=device)

    assert same(expected_seq_likelihoods_for_each_provider, seq_likelihoods_for_each_provider, eps=1e-2)

    expected_seq_likelihoods_by_context = torch.tensor(
        [[[[2.1000, 0.9000, 0.0000],
           [0.0000, 3.0000, 0.0000]],
          [[0.3000, 2.7000, 0.0000],
           [0.0000, 3.0000, 0.0000]],
          [[0.0000, 0.0000, 0.0000],
           [0.0000, 0.0000, 0.0000]]],
         [[[0.8000, 0.6000, 0.6000],
           [2.0000, 0.0000, 0.0000]],
          [[0.2000, 0.6000, 0.0000],
           [1.0000, 0.0000, 0.0000]],
          [[0.2000, 0.8000, 0.0000],
           [0.0000, 0.0000, 1.0000]]]],

        dtype=float_dtype, device=device)
    assert same(expected_seq_likelihoods_by_context, seq_likelihoods_by_context, eps=1e-4)


def test_disambiguating_provider_informativeness():
    device = 'cuda'
    float_dtype = get_float(device)
    flock_size = 1
    n_cluster_centers = 3
    n_frequent_seqs = 2
    seq_lookahead = 1
    seq_length = 3
    seq_lookbehind = seq_length - seq_lookahead
    buffer_size = 3
    context_size = 3
    n_providers = 2
    nan = FLOAT_NAN

    buffer = create_tp_buffer(flock_size=flock_size, buffer_size=buffer_size, n_cluster_centers=n_cluster_centers,
                              n_frequent_seqs=n_frequent_seqs, context_size=context_size, device=device,
                              n_providers=n_providers)

    buffer.contexts.stored_data = torch.tensor([[[[0, 1, nan], [1, 0, 0]],
                                                 [[1, 0, nan], [0, 0, 1]],
                                                 [[1, 0, nan], [0, 1, 0]]]], dtype=float_dtype, device=device)

    process = create_tp_flock_forward_process(flock_size=flock_size, n_cluster_centers=n_cluster_centers,
                                              n_frequent_seqs=n_frequent_seqs, seq_lookahead=seq_lookahead,
                                              seq_length=seq_length, context_size=context_size, buffer=buffer,
                                              n_providers=n_providers,
                                              device=device)

    context_history = torch.zeros((flock_size, seq_lookbehind, n_providers, context_size),
                                  dtype=float_dtype, device=device)

    seq_likelihoods_priors_clusters = torch.tensor([[1, 1]], dtype=float_dtype, device=device)

    # Assuming context prior of 5, and seeing each sequence 15 times
    # p1 sees [[1, 0, nan], [1, 0, nan], [1, 0, nan]] for seq 1 and [[1, 0, nan], [0, 1, nan], [1, 0, nan]] for seq 2
    # P2 sees [[0, 0, 1], [1, 0, 0], [0, 1, 0] for both sequences
    # P1 is padded up to the size of P2
    frequent_context_likelihoods = torch.tensor([[[[[0.8, 0.2, nan], [0.2, 0.2, 0.8]],
                                                [[0.8, 0.2, nan], [0.2, 0.8, 0.2]],
                                                [[0.8, 0.2, nan], [0.8, 0.2, 0.2]]],

                                               [[[0.8, 0.2, nan], [0.2, 0.2, 0.8]],
                                                [[0.2, 0.8, nan], [0.2, 0.8, 0.2]],
                                                [[0.8, 0.2, nan], [0.8, 0.2, 0.2]]]]], dtype=float_dtype, device=device)

    seq_likelihoods_for_each_provider = torch.zeros((flock_size, n_frequent_seqs, n_providers, seq_lookbehind),
                                                    dtype=float_dtype, device=device)

    seq_likelihoods_by_context = torch.zeros(flock_size, n_frequent_seqs, n_providers, context_size, dtype=float_dtype,
                                             device=device)

    process._compute_seq_likelihoods_for_each_provider(buffer, context_history, seq_likelihoods_priors_clusters,
                                                       frequent_context_likelihoods,
                                                       seq_likelihoods_for_each_provider, seq_likelihoods_by_context)

    # Assert that the seq_likelihoods are correct
    expected_seq_likelihoods_for_each_provider = torch.tensor([[[[0.8, 0.8], [0.8, 0.8]], [[0.8, 0.8], [0.2, 0.8]]]],
                                                              dtype=float_dtype, device=device)
    assert same(expected_seq_likelihoods_for_each_provider, seq_likelihoods_for_each_provider, eps=1e-4)

    expected_seq_likelihoods_by_context = torch.tensor(
        [[[[0.8000, 0.2000, 0.0000], [0.8000, 0.2000, 0.2000]], [[0.8000, 0.2000, 0.0000], [0.8000, 0.2000, 0.2000]]]],
        dtype=float_dtype, device=device)
    assert same(expected_seq_likelihoods_by_context, seq_likelihoods_by_context)

    provider_informativeness = torch.zeros((flock_size, n_providers, seq_lookbehind), dtype=float_dtype, device=device)

    process._compute_provider_informativeness(seq_likelihoods_priors_clusters, seq_likelihoods_for_each_provider,
                                              provider_informativeness)

    # Assert the informativeness is correct
    expected_provider_informativeness = torch.tensor([[[0., 0.], [0.2231, 0.]]], dtype=float_dtype, device=device)
    assert same(expected_provider_informativeness, provider_informativeness, eps=1e-4)

    # Now we assume that we have seen the sequences 800 times...
    frequent_context_likelihoods = torch.tensor([[[[[0.993, 0.006, nan], [0.006, 0.006, 0.993]],
                                                [[0.993, 0.006, nan], [0.006, 0.993, 0.006]],
                                                [[0.993, 0.006, nan], [0.993, 0.006, 0.006]]],

                                               [[[0.993, 0.006, nan], [0.006, 0.006, 0.993]],
                                                [[0.006, 0.993, nan], [0.006, 0.993, 0.006]],
                                                [[0.993, 0.006, nan], [0.993, 0.006, 0.006]]]]], dtype=float_dtype,
                                             device=device)

    process._compute_seq_likelihoods_for_each_provider(buffer, context_history, seq_likelihoods_priors_clusters,
                                                       frequent_context_likelihoods,
                                                       seq_likelihoods_for_each_provider, seq_likelihoods_by_context)

    # Assert that the seq_likelihoods are correct
    expected_seq_likelihoods_for_each_provider = torch.tensor(
        [[[[0.9930, 0.9930], [0.9930, 0.9930]], [[0.9930, 0.9930], [0.006, 0.9930]]]], dtype=float_dtype, device=device)
    assert same(expected_seq_likelihoods_for_each_provider, seq_likelihoods_for_each_provider, eps=1e-4)

    expected_seq_likelihoods_by_context = torch.tensor(
        [[[[0.9930, 0.0060, 0.0000], [0.9930, 0.0060, 0.0060]], [[0.9930, 0.0060, 0.0000], [0.9930, 0.0060, 0.0060]]]],
        dtype=float_dtype, device=device)
    assert same(expected_seq_likelihoods_by_context, seq_likelihoods_by_context)

    process._compute_provider_informativeness(seq_likelihoods_priors_clusters, seq_likelihoods_for_each_provider,
                                              provider_informativeness)

    # Assert the informativeness is correct
    expected_provider_informativeness = torch.tensor([[[0., 0.], [1.8674, 0.]]], dtype=float_dtype, device=device)
    assert same(expected_provider_informativeness, provider_informativeness, eps=1e-4)


def test_compute_provider_informativeness():
    device = 'cuda'
    float_dtype = get_float(device)
    flock_size = 2
    n_cluster_centers = 4
    n_frequent_seqs = 3
    seq_lookahead = 1
    seq_lookbehind = 2
    seq_length = 3
    buffer_size = 3
    context_size = 3
    n_providers = 2

    buffer = create_tp_buffer(flock_size=flock_size, buffer_size=buffer_size, n_cluster_centers=n_cluster_centers,
                              n_frequent_seqs=n_frequent_seqs, context_size=context_size, device=device)

    process = create_tp_flock_forward_process(flock_size=flock_size, n_cluster_centers=n_cluster_centers,
                                              n_frequent_seqs=n_frequent_seqs, seq_lookahead=seq_lookahead,
                                              seq_length=seq_length, context_size=context_size, buffer=buffer,
                                              device=device)

    context_informativeness = torch.zeros((flock_size, seq_lookbehind, n_providers), dtype=float_dtype, device=device)

    # From the previous tests...
    seq_likelihoods_priors_clusters = torch.tensor([[4, 3, 0], [2, 3, 1]], dtype=float_dtype, device=device)

    # Result from the previous test...
    seq_likelihoods_for_each_provider = torch.tensor([[[[1.5, 3], [0, 0]],
                                                       [[4.8, 3], [0, 0]],
                                                       [[0, 0], [0, 0]]],
                                                      [[[1.6, 0.1], [0, 0]],
                                                       [[0.7, 0.1], [0, 0]],
                                                       [[1.3, 1], [0, 0]]]], dtype=float_dtype, device=device)

    process._compute_provider_informativeness(seq_likelihoods_priors_clusters, seq_likelihoods_for_each_provider,
                                              context_informativeness)

    expected_context_informativeness = torch.tensor([[[0.2537, 0.0102], [0., 0.]], [[0.2475, 1.0897], [0., 0.]]],
                                                    dtype=float_dtype,
                                                    device=device)

    assert same(expected_context_informativeness, context_informativeness, eps=1e-3)


def test_compute_seq_likelihoods_priors_clusters_context():
    device = 'cuda'
    float_dtype = get_float(device)
    flock_size = 2
    n_cluster_centers = 4
    n_frequent_seqs = 3
    seq_lookahead = 1
    seq_length = 3
    buffer_size = 3
    context_size = 2

    buffer = create_tp_buffer(flock_size=flock_size, buffer_size=buffer_size, n_cluster_centers=n_cluster_centers,
                              n_frequent_seqs=n_frequent_seqs, context_size=context_size, device=device)

    process = create_tp_flock_forward_process(flock_size=flock_size, n_cluster_centers=n_cluster_centers,
                                              n_frequent_seqs=n_frequent_seqs, seq_lookahead=seq_lookahead,
                                              seq_length=seq_length, context_size=context_size, buffer=buffer,
                                              device=device)

    seq_likelihoods_priors_clusters_context = torch.zeros((flock_size, n_frequent_seqs), dtype=float_dtype,
                                                          device=device)

    # Result from the previous test...
    context_informativeness = torch.tensor([[[0.9621, 0.2674], [0.0017, 0.1186]],
                                            [[0.9939, 1.3206], [0.0865, 0.7034]]], dtype=float_dtype, device=device)

    # From previous tests...
    seq_likelihoods_for_each_context = torch.tensor([[[[0.3, 1.8], [4.5, 0.3]],
                                                      [[4.8, 6], [3, 0.6]],
                                                      [[0, 0], [0, 0]]],
                                                     [[[4, 6], [1, 5.2]],
                                                      [[0.3, 0.2], [0.6, 0.6]],
                                                      [[0.1, 0.2], [0.3, 0.3]]]], dtype=float_dtype, device=device)

    process._compute_seq_likelihoods_priors_clusters_context(seq_likelihoods_priors_clusters_context,
                                                             seq_likelihoods_for_each_context, context_informativeness)

    expected_seq_likelihoods_priors_clusters_context = torch.tensor([[0.3, 4.8, 0], [6, 0.2, 0.2]], dtype=float_dtype,
                                                                    device=device)

    assert same(expected_seq_likelihoods_priors_clusters_context, seq_likelihoods_priors_clusters_context)


def test_compute_seq_probs_without_priors():
    device = 'cuda'
    float_dtype = get_float(device)
    flock_size = 2
    n_frequent_seqs = 3

    # Preallocate by some value to chach that it was overwritten
    seq_probs_clusters_context = torch.full((flock_size, n_frequent_seqs), fill_value=-2, dtype=float_dtype,
                                            device=device)

    # Result from the previous test...
    seq_likelihoods_priors_clusters_context = torch.tensor([[0.3, 4.8, 0], [6, 0.2, 0.2]], dtype=float_dtype,
                                                           device=device)

    # From previous tests...
    frequent_seq_occurrences = torch.tensor([[2, 3, 0], [4, 1, 1]], dtype=float_dtype, device=device)

    TPFlockForwardAndBackward._compute_seq_probs_without_priors(seq_likelihoods_priors_clusters_context,
                                                                frequent_seq_occurrences,
                                                                seq_probs_clusters_context)

    expected_seq_probs_clusters_context = torch.tensor([[0.0857, 0.9143, 0],
                                                        [0.7895, 0.1053, 0.1053]], dtype=float_dtype, device=device)

    assert same(expected_seq_probs_clusters_context, seq_probs_clusters_context, eps=1e-4)


# endregion


def test_apply_output_projection():
    device = 'cuda'
    float_dtype = get_float(device)

    flock_size = 2
    n_frequent_seqs = 4
    n_cluster_centers = 4
    seq_length = 3
    seq_lookahead = 1

    # region Setup

    buffer = create_tp_buffer(flock_size=flock_size, n_frequent_seqs=n_frequent_seqs, device=device)

    frequent_seqs = torch.tensor([[[0, 1, 2],
                                   [1, 2, 3],
                                   [2, 3, 0],
                                   [3, 0, 1]],
                                  [[0, 1, 0],
                                   [1, 0, 1],
                                   [0, 1, 2],
                                   [1, 2, 0]]], dtype=torch.int64, device=device)

    frequent_occurrences = torch.tensor([[3, 2, 1, 0],
                                         [10, 2, 2, 0.1]], dtype=float_dtype, device=device)

    outputs = torch.full((flock_size, n_cluster_centers), fill_value=-1, dtype=float_dtype, device=device)

    process = create_tp_flock_forward_process(buffer=buffer,
                                              frequent_seqs=frequent_seqs,
                                              frequent_seq_occurrences=frequent_occurrences,
                                              n_frequent_seqs=n_frequent_seqs,
                                              n_cluster_centers=n_cluster_centers,
                                              seq_length=seq_length,
                                              seq_lookahead=seq_lookahead,
                                              device=device)

    process._seq_likelihoods = torch.tensor([[0, 0, 2, 0],
                                             [0.5, 0, 0.5, 0]], dtype=float_dtype, device=device)

    process._apply_output_projection(frequent_seqs,
                                     process._seq_likelihoods,
                                     outputs)

    expected_outputs = torch.tensor([[0.25, 0, 0.25, 0.5],
                                     [0.375, 0.5, 0.125, 0]], dtype=float_dtype, device=device)

    assert same(expected_outputs, outputs)


def test_calculate_predicted_clusters():
    device = 'cuda'
    float_dtype = get_float(device)

    flock_size = 2
    n_frequent_seqs = 5
    n_cluster_centers = 4
    seq_length = 4

    # region Setup

    frequent_seqs = torch.tensor([[[0, 1, 2, 3], [0, 2, 3, 2], [1, 3, 2, 0], [3, 2, 3, 2], [-1, -1, -1, -1]],
                                  [[1, 3, 2, 1], [0, 1, 0, 3], [3, 2, 0, 1], [1, 3, 2, 3], [3, 2, 1, 2]]],
                                 dtype=torch.int64,
                                 device=device)

    process1 = create_tp_flock_forward_process(flock_size=flock_size,
                                               n_frequent_seqs=n_frequent_seqs,
                                               n_cluster_centers=n_cluster_centers,
                                               seq_length=seq_length,
                                               frequent_seqs=frequent_seqs,
                                               device=device)

    process2 = create_tp_flock_forward_process(flock_size=flock_size,
                                               n_frequent_seqs=n_frequent_seqs,
                                               n_cluster_centers=n_cluster_centers,
                                               seq_length=seq_length,
                                               seq_lookahead=2,
                                               frequent_seqs=frequent_seqs,
                                               device=device)

    pred_outputs1 = torch.zeros(flock_size, seq_length, n_cluster_centers, dtype=float_dtype, device=device)
    pred_outputs2 = torch.zeros(flock_size, seq_length, n_cluster_centers, dtype=float_dtype, device=device)

    seq_likelihoods1 = torch.tensor([[3, 1, 2, 2, 0], [4, 1, 2, 4, 0]], dtype=float_dtype, device=device)
    seq_likelihoods2 = torch.tensor([[2, 1, 1, 1, 0], [2, 3, 1, 2, 0]], dtype=float_dtype, device=device)

    expected_pred_outputs1 = torch.tensor([[[0.5, 0.25, 0, 0.25],
                                            [0, 0.375, 0.375, 0.25],
                                            [0, 0, 0.625, 0.375],
                                            [0.25, 0, 0.375, 0.375]],
                                           [[0.0909, 0.7273, 0, 0.1818],
                                            [0, 0.0909, 0.1818, 0.7273],
                                            [0.2727, 0, 0.7273, 0],
                                            [0.0000, 0.5455, 0.0000, 0.4545]]], dtype=float_dtype, device=device)

    expected_pred_outputs2 = torch.tensor([[[0.6, 0.2, 0, 0.2],
                                            [0, 0.4, 0.4, 0.2],
                                            [0, 0, 0.6, 0.4],
                                            [0.2, 0, 0.4, 0.4]],
                                           [[0.375, 0.5, 0, 0.125],
                                            [0, 0.375, 0.125, 0.5],
                                            [0.5, 0, 0.5, 0],
                                            [0, 0.3750, 0, 0.6250]]], dtype=float_dtype, device=device)

    # endregion

    process1._compute_predicted_clusters(frequent_seqs, seq_likelihoods1, pred_outputs1)
    process2._compute_predicted_clusters(frequent_seqs, seq_likelihoods2, pred_outputs2)

    assert same(expected_pred_outputs1, pred_outputs1, eps=1e-3)
    assert same(expected_pred_outputs2, pred_outputs2, eps=1e-3)


def test_compute_rewards():
    """ Tests that the rewards are correctly computed for the lookaheads of frequent sequences.

    This tests both a 'higher level' which only computes rewards for the transitions after the current one, and a
    base level with computes rewards for all future transitions.
    """
    device = 'cuda'
    float_dtype = get_float(device)

    flock_size = 1
    n_frequent_seqs = 3
    n_cluster_centers = 4
    seq_length = 3
    seq_lookahead = 2
    context_size = 3
    n_providers = 2

    # region Setup

    buffer = create_tp_buffer(flock_size=flock_size, n_frequent_seqs=n_frequent_seqs,
                              context_size=context_size, device=device)

    frequent_seqs = torch.tensor([[[0, 1, 2],
                                   [0, 1, 3],
                                   [0, 2, 3]]], dtype=torch.int64, device=device)

    freq_seq_occurrences = torch.tensor([[0, 0, 1]], dtype=torch.int64, device=device)

    context_data = torch.tensor([[[[0, 0, 0],  # The current context doesn't matter for this test
                                   [5, 3, 0],  # P1 Rewards
                                   [0, 0, 0]],  # P1 Punishments

                                  [[0, 0, 0],
                                   [1.5, 3, 0],  # P2 Rewards
                                   [0, 0, 0]]]],  # P2 Punishments
                                dtype=float_dtype, device=device)

    frequent_context_likelihoods = torch.tensor([[[[[0.7, 0.2, 0.1],  # P1
                                                 [0.5, 0.3, 0.2]],  # P2

                                                [[0.6, 0.2, 0.5],  # P1
                                                 [0.5, 0.2, 0.1]],  # P2

                                                [[0.4, 0, 0.1],  # P1
                                                 [0.6, 0.7, 0.5]]],  # P2

                                               [[[0.3, 0.5, 0.7],  # P1
                                                 [0.9, 0.2, 0.3]],  # P2

                                                [[0.1, 0.2, 0.5],  # P1
                                                 [0.3, 0.4, 0.5]],  # P2

                                                [[0.7, 0.4, 0.2],  # P1
                                                 [0.5, 0.5, 0]]],  # P2

                                               [[[0.9, 0.1, 0.3],  # P1
                                                 [0.2, 0, 0.1]],  # P2

                                                [[0.25, 0, 0.5],  # P1
                                                 [0, 0.3, 0]],  # P2

                                                [[1, 1, 0],  # P1
                                                 [1, 0, 1]]]]],  # P2
                                             dtype=float_dtype, device=device)

    # The 0.01 values here test for if the model correctly handles nonzero transition probabilities with
    # no corresponding transitions contained in frequent_seqs
    frequent_exploration_destination_dists = torch.tensor([[[[0.01, 0.7, 0.3, 0.01],  # T1
                                                             [0.01, 0.01, 0.6, 0.4]],  # T2

                                                            [[0.01, 0.7, 0.3, 0.01],  # T1
                                                             [0.01, 0.01, 0.1, 0.9]],  # T2

                                                            [[0.01, 0.2, 0.8, 0.01],  # T1
                                                             [0.01, 0.01, 0.01, 1]]]],  # T2
                                                          dtype=float_dtype, device=device)

    seq_probs_clusters_context = torch.tensor([[0.2, 0.3, 0.5]], dtype=float_dtype, device=device)

    process1 = create_tp_flock_forward_process(flock_size=flock_size,
                                               buffer=buffer,
                                               n_frequent_seqs=n_frequent_seqs,
                                               n_cluster_centers=n_cluster_centers,
                                               seq_length=seq_length,
                                               seq_lookahead=seq_lookahead,
                                               context_size=context_size,
                                               n_providers=n_providers,
                                               device=device,
                                               produce_actions=False)

    seq_rewards_goal_directed1 = torch.zeros((flock_size, n_frequent_seqs, 2),
                                             dtype=float_dtype, device=device)

    expected_seq_rewards1 = torch.tensor([[[0.720, 0], [1.1961, 0], [3.6, 0]]], dtype=float_dtype, device=device)

    frequent_rewards_punishments = torch.zeros((flock_size, n_frequent_seqs, seq_lookahead, 2), dtype=float_dtype, device=device)

    process1._compute_rewards(frequent_seqs,
                              freq_seq_occurrences,
                              context_data,
                              frequent_context_likelihoods,
                              seq_probs_clusters_context,
                              frequent_rewards_punishments,
                              frequent_exploration_destination_dists,
                              seq_rewards_goal_directed1)

    assert same(expected_seq_rewards1, seq_rewards_goal_directed1, eps=1e-3)

    # Calculating both transitions
    process2 = create_tp_flock_forward_process(flock_size=flock_size,
                                               buffer=buffer,
                                               n_frequent_seqs=n_frequent_seqs,
                                               n_cluster_centers=n_cluster_centers,
                                               seq_length=seq_length,
                                               seq_lookahead=seq_lookahead,
                                               context_size=context_size,
                                               n_providers=n_providers,
                                               device=device,
                                               produce_actions=True)

    seq_rewards_goal_directed2 = torch.zeros((flock_size, n_frequent_seqs, 2),
                                             dtype=float_dtype, device=device)

    expected_seq_rewards2 = torch.tensor([[[0.8423, 0], [1.3366, 0], [2.9368, 0]]], dtype=float_dtype, device=device)

    process2._compute_rewards(frequent_seqs,
                              freq_seq_occurrences,
                              context_data,
                              frequent_context_likelihoods,
                              seq_probs_clusters_context,
                              frequent_rewards_punishments,
                              frequent_exploration_destination_dists,
                              seq_rewards_goal_directed2)

    assert same(expected_seq_rewards2, seq_rewards_goal_directed2, eps=1e-3)


def test_compute_rewards2():
    """ Tests that the rewards are correctly computed for the lookaheads of frequent sequences.

    This tests both a 'higher level' which only computes rewards for the transitions after the current one, and a
    base level with computes rewards for all future transitions.

    This is the same as test_compute_rewards1, but with a different 'topology of sequences'
    """
    device = 'cuda'
    float_dtype = get_float(device)

    flock_size = 1
    n_frequent_seqs = 4
    n_cluster_centers = 4
    seq_length = 3
    seq_lookahead = 2
    context_size = 3
    n_providers = 2

    # region Setup

    buffer = create_tp_buffer(flock_size=flock_size, n_frequent_seqs=n_frequent_seqs,
                              context_size=context_size, device=device)

    frequent_seqs = torch.tensor([[[0, 1, 2],
                                   [0, 2, 3],
                                   [1, 2, 3],
                                   [0, 1, 3]]], dtype=torch.int64, device=device)

    freq_seq_occurrences = torch.tensor([[0, 0, 1, 0]], dtype=torch.int64, device=device)

    context_data = torch.tensor([[[[0, 0, 0],  # The current context doesn't matter for this test
                                   [5, 3, 1.2],  # P1 Rewards
                                   [0, 0, 0]],  # P1 Punishments

                                  [[0, 0, 0],
                                   [1.5, 7, 2],  # P2 Rewards
                                   [0, 0, 0]]]],  # P2 Punishments
                                dtype=float_dtype, device=device)

    frequent_context_likelihoods = torch.tensor([[[[[0.7, 0.2, 0.1],  # P1
                                                 [0.5, 0.3, 0.2]],  # P2

                                                [[0.6, 0.2, 0.5],  # P1
                                                 [0.5, 0.2, 0.1]],  # P2

                                                [[0.4, 0, 0.1],  # P1
                                                 [0.6, 0.7, 0.5]]],  # P2

                                               [[[0.3, 0.5, 0.7],  # P1
                                                 [0.9, 0.2, 0.3]],  # P2

                                                [[0.1, 0.2, 0.5],  # P1
                                                 [0.3, 0.4, 0.5]],  # P2

                                                [[0.7, 0.4, 0.2],  # P1
                                                 [0.5, 0.5, 0]]],  # P2

                                               [[[0.9, 0.1, 0.3],  # P1
                                                 [0.2, 0, 0.1]],  # P2

                                                [[0.25, 0, 0.5],  # P1
                                                 [0, 0.3, 0]],  # P2

                                                [[0.5, 0.2, 0],  # P1
                                                 [0.3, 0, 0.6]]],

                                               [[[0.2, 0.7, 0.1],  # P1
                                                 [0.8, 0.1, 0.2]],  # P2

                                                [[0.4, 0.2, 0.9],  # P1
                                                 [0.1, 0.3, 0.1]],  # P2

                                                [[0.5, 0.2, 0.6],  # P1
                                                 [0.3, 0.1, 0.3]]]]],  # P2
                                             dtype=float_dtype, device=device)

    frequent_exploration_destination_dists = torch.tensor([[[[0, 0.6, 0.4, 0],  # T1
                                                             [0, 0, 0.8, 0.2]],  # T2

                                                            [[0, 0.2, 0.8, 0],  # T1
                                                             [0, 0, 0, 1]],  # T2

                                                            [[0, 0, 1, 0],  # T1
                                                             [0, 0, 0, 1]],

                                                            [[0, 0.9, 0.1, 0],  # T1
                                                             [0, 0, 0.3, 0.7]]]],  # T2
                                                          dtype=float_dtype, device=device)

    seq_probs_clusters_context = torch.tensor([[0.2, 0.25, 0.35, 0.2]], dtype=float_dtype, device=device)

    process1 = create_tp_flock_forward_process(flock_size=flock_size,
                                               buffer=buffer,
                                               n_frequent_seqs=n_frequent_seqs,
                                               n_cluster_centers=n_cluster_centers,
                                               seq_length=seq_length,
                                               seq_lookahead=seq_lookahead,
                                               context_size=context_size,
                                               n_providers=n_providers,
                                               device=device,
                                               produce_actions=False)

    seq_rewards_goal_directed1 = torch.zeros((flock_size, n_frequent_seqs, 2),
                                             dtype=float_dtype, device=device)

    expected_seq_rewards1 = torch.tensor([[[1.0422, 0], [1.1115, 0], [0.9765, 0], [0.736, 0]]], dtype=float_dtype,
                                         device=device)

    frequent_rewards_punishments = torch.zeros((flock_size, n_frequent_seqs, seq_lookahead, 2), dtype=float_dtype, device=device)

    process1._compute_rewards(frequent_seqs,
                              freq_seq_occurrences,
                              context_data,
                              frequent_context_likelihoods,
                              seq_probs_clusters_context,
                              frequent_rewards_punishments,
                              frequent_exploration_destination_dists,
                              seq_rewards_goal_directed1)

    assert same(expected_seq_rewards1, seq_rewards_goal_directed1, eps=1e-3)

    # Calculating both transitions
    process2 = create_tp_flock_forward_process(flock_size=flock_size,
                                               buffer=buffer,
                                               n_frequent_seqs=n_frequent_seqs,
                                               n_cluster_centers=n_cluster_centers,
                                               seq_length=seq_length,
                                               seq_lookahead=seq_lookahead,
                                               context_size=context_size,
                                               n_providers=n_providers,
                                               device=device,
                                               produce_actions=True)

    seq_rewards_goal_directed2 = torch.zeros((flock_size, n_frequent_seqs, 2),
                                             dtype=float_dtype, device=device)

    expected_seq_rewards2 = torch.tensor([[[0.8686, 0], [0.9775, 0], [0.8788, 0], [0.6762, 0]]], dtype=float_dtype,
                                         device=device)

    process2._compute_rewards(frequent_seqs,
                              freq_seq_occurrences,
                              context_data,
                              frequent_context_likelihoods,
                              seq_probs_clusters_context,
                              frequent_rewards_punishments,
                              frequent_exploration_destination_dists,
                              seq_rewards_goal_directed2)

    assert same(expected_seq_rewards2, seq_rewards_goal_directed2, eps=1e-3)


def test_compute_rewards_closest_reward():
    """Tests to see that the reward that is closest (all other things being equal) is the reward that the
     agent should go towards."""
    device = 'cuda'
    float_dtype = get_float(device)

    flock_size = 1
    n_frequent_seqs = 3
    n_cluster_centers = 5
    seq_length = 4
    seq_lookahead = 3
    context_size = 3
    n_providers = 1

    # region Setup

    buffer = create_tp_buffer(flock_size=flock_size, n_frequent_seqs=n_frequent_seqs,
                              context_size=context_size, device=device)

    frequent_seqs = torch.tensor([[[0, 4, 1, 3],
                                   [0, 1, 3, 2],
                                   [0, 2, 4, 1]]], dtype=torch.int64, device=device)

    freq_seq_occurrences = torch.tensor([[5, 5, 5]], dtype=torch.int64, device=device)

    context_data = torch.tensor([[[[0, 0, 0],  # The current context doesn't matter for this test
                                   [0, 0, 0],  # P1 Rewards
                                   [0, 0, 0]]]],  # P2 Punishments
                                dtype=float_dtype, device=device)

    frequent_context_likelihoods = torch.zeros(flock_size, n_frequent_seqs, seq_length, n_providers, context_size,
                                               dtype=float_dtype, device=device)

    frequent_exploration_destination_dists = torch.tensor([[[[0, 0, 0, 0, 1],  # T1
                                                             [0, 1, 0, 0, 0],
                                                             [0, 0, 0, 1, 0]],  # T2

                                                            [[0, 1, 0, 0, 0],  # T1
                                                             [0, 0, 0, 1, 0],
                                                             [0, 0, 1, 0, 0]],  # T2

                                                            [[0, 0, 1, 0, 0],  # T1
                                                             [0, 0, 0, 0, 1],
                                                             [0, 1, 0, 0, 0]]]],  # T2
                                                          dtype=float_dtype, device=device)

    seq_probs_clusters_context = torch.tensor([[0.33, 0.33, 0.33]], dtype=float_dtype, device=device)

    frequent_rewards_punishments = torch.tensor([[[[0, 0], [25, 0], [0, 0]],
                                                  [[25, 0], [0, 0], [0, 0]],
                                                  [[0, 0], [0, 0], [25, 0]]]], dtype=float_dtype, device=device)

    process = create_tp_flock_forward_process(flock_size=flock_size,
                                               buffer=buffer,
                                               n_frequent_seqs=n_frequent_seqs,
                                               n_cluster_centers=n_cluster_centers,
                                               seq_length=seq_length,
                                               seq_lookahead=seq_lookahead,
                                               context_size=context_size,
                                               n_providers=n_providers,
                                               device=device,
                                               produce_actions=True)

    process.own_rewards_weight = 1

    seq_rewards_goal_directed = torch.zeros((flock_size, n_frequent_seqs, 2),
                                             dtype=float_dtype, device=device)

    expected_seq_rewards = torch.tensor([[[1.3365, 0], [1.485, 0], [1.2028, 0]]], dtype=float_dtype,
                                         device=device)

    process._compute_rewards(frequent_seqs,
                              freq_seq_occurrences,
                              context_data,
                              frequent_context_likelihoods,
                              seq_probs_clusters_context,
                              frequent_rewards_punishments,
                              frequent_exploration_destination_dists,
                              seq_rewards_goal_directed)

    assert same(expected_seq_rewards, seq_rewards_goal_directed, eps=1e-3)

def test_compute_rewards_with_scaled_likelihoods_and_expert_rewards():
    """ Tests that the rewards are correctly computed for the lookaheads of frequent sequences.

    This tests both a 'higher level' which only computes rewards for the transitions after the current one, and a
    base level with computes rewards for all future transitions.

    This is similar to test_compute rewards2, but with a different 'topology of sequences' and one of the sequences
    has a huge reward, but is also very unlikely. The expected value of the 'falling' into this sequence from another
    should be scaled by how likely the sequence is.

    This test also has some rewards for the current expert which should be divided by their occurrences ,
    then multiplied by the default 0.1 scaling value for its own rewards which will influence the results by a relatively small amount
    reward for each parent.
    """
    device = 'cuda'
    float_dtype = get_float(device)

    flock_size = 1
    n_frequent_seqs = 4
    n_cluster_centers = 5
    seq_length = 3
    seq_lookahead = 2
    context_size = 3
    n_providers = 2

    # region Setup

    buffer = create_tp_buffer(flock_size=flock_size, n_frequent_seqs=n_frequent_seqs,
                              context_size=context_size, device=device)

    frequent_seqs = torch.tensor([[[0, 1, 2],
                                   [0, 1, 3],
                                   [0, 1, 4],
                                   [0, 2, 3]]], dtype=torch.int64, device=device)

    freq_seq_occurrences = torch.tensor([[5, 3, 2, 1]], dtype=torch.int64, device=device)

    context_data = torch.tensor([[[[0, 0, 0],  # The current context doesn't matter for this test
                                   [5, 3, 1.2],  # P1 Rewards
                                   [0, 0, 0]],  # P1 Punishments

                                  [[0, 0, 0],
                                   [1.5, 7, 1000],  # P2 Rewards
                                   [0, 0, 0]]]],  # P2 Punishments
                                dtype=float_dtype, device=device)

    frequent_context_likelihoods = torch.tensor([[[[[0.7, 0.2, 0.1],  # P1
                                                 [0.5, 0.3, 0.0]],  # P2

                                                [[0.6, 0.2, 0.5],  # P1
                                                 [0.5, 0.2, 0.0]],  # P2

                                                [[0.4, 0, 0.1],  # P1
                                                 [0.6, 0.7, 0.0]]],  # P2

                                               [[[0.3, 0.5, 0.7],  # P1
                                                 [0.9, 0.2, 0.0]],  # P2

                                                [[0.1, 0.2, 0.5],  # P1
                                                 [0.3, 0.4, 0.0]],  # P2

                                                [[0.7, 0.4, 0.2],  # P1
                                                 [0.5, 0.5, 0]]],  # P2

                                               [[[0.9, 0.1, 0.3],  # P1
                                                 [0.2, 0, 0]],  # P2

                                                [[0.25, 0, 0.5],  # P1
                                                 [0, 0.3, 0]],  # P2

                                                [[0.5, 0.2, 0],  # P1
                                                 [0.3, 0, 1]]],

                                               [[[0.3, 0.5, 0.7],  # P1
                                                 [0.9, 0.2, 0.0]],  # P2

                                                [[0.1, 0.2, 0.5],  # P1
                                                 [0.3, 0.4, 0.0]],  # P2

                                                [[0.7, 0.4, 0.2],  # P1
                                                 [0.5, 0.5, 0]]]]],  # P2
                                             dtype=float_dtype, device=device)

    frequent_exploration_destination_dists = torch.tensor([[[[0, 1, 0, 0, 0],  # T1
                                                             [0, 0, 1, 0, 0]],  # T2

                                                            [[0, 1, 0, 0, 0],  # T1
                                                             [0, 0, 0, 1, 0]],  # T2

                                                            [[0, 1, 0, 0, 0],  # T1
                                                             [0, 0, 0, 0, 1]], # T2

                                                            [[0, 0.5, 0.5, 0, 0],  # T1
                                                             [0, 0, 0, 1, 0]]]],  # T2
                                                          dtype=float_dtype, device=device)

    seq_probs_clusters_context = torch.tensor([[0.21, 0.34, 0.0001, 0.4499]], dtype=float_dtype, device=device)

    process1 = create_tp_flock_forward_process(flock_size=flock_size,
                                               buffer=buffer,
                                               n_frequent_seqs=n_frequent_seqs,
                                               n_cluster_centers=n_cluster_centers,
                                               seq_length=seq_length,
                                               seq_lookahead=seq_lookahead,
                                               context_size=context_size,
                                               n_providers=n_providers,
                                               device=device,
                                               produce_actions=False)

    seq_rewards_goal_directed1 = torch.zeros((flock_size, n_frequent_seqs, 2),
                                             dtype=float_dtype, device=device)

    expected_seq_rewards1 = torch.tensor([[[1.1529, 0], [1.5422, 0], [0.09, 0], [2.0003, 0]]], dtype=float_dtype,
                                         device=device)

    frequent_rewards_punishments = torch.tensor([[[[10, 0], [15, 0]],
                                                  [[3.3, 0], [3, 0]],
                                                  [[2, 0], [2, 0]],
                                                  [[0, 0], [0, 0]]]], dtype=float_dtype, device=device)

    process1._compute_rewards(frequent_seqs,
                              freq_seq_occurrences,
                              context_data,
                              frequent_context_likelihoods,
                              seq_probs_clusters_context,
                              frequent_rewards_punishments,
                              frequent_exploration_destination_dists,
                              seq_rewards_goal_directed1)

    assert same(expected_seq_rewards1, seq_rewards_goal_directed1, eps=1e-3)

    # Calculating both transitions
    process2 = create_tp_flock_forward_process(flock_size=flock_size,
                                               buffer=buffer,
                                               n_frequent_seqs=n_frequent_seqs,
                                               n_cluster_centers=n_cluster_centers,
                                               seq_length=seq_length,
                                               seq_lookahead=seq_lookahead,
                                               context_size=context_size,
                                               n_providers=n_providers,
                                               device=device,
                                               produce_actions=True)

    seq_rewards_goal_directed2 = torch.zeros((flock_size, n_frequent_seqs, 2),
                                             dtype=float_dtype, device=device)

    expected_seq_rewards2 = torch.tensor([[[1.0376, 0], [1.3880, 0], [.0810, 0], [1.8077, 0]]], dtype=float_dtype,
                                         device=device)

    process2._compute_rewards(frequent_seqs,
                              freq_seq_occurrences,
                              context_data,
                              frequent_context_likelihoods,
                              seq_probs_clusters_context,
                              frequent_rewards_punishments,
                              frequent_exploration_destination_dists,
                              seq_rewards_goal_directed2)

    assert same(expected_seq_rewards2, seq_rewards_goal_directed2, eps=1e-3)


def test_compute_predicted_clusters():
    device = 'cuda'
    float_dtype = get_float(device)

    flock_size = 2
    n_frequent_sequences = 4
    n_cluster_centers = 4
    seq_length = 4

    # region Setup
    process1 = create_tp_flock_forward_process(flock_size=flock_size,
                                               n_frequent_seqs=n_frequent_sequences,
                                               n_cluster_centers=n_cluster_centers,
                                               seq_length=seq_length,
                                               device=device)

    process2 = create_tp_flock_forward_process(flock_size=flock_size,
                                               n_frequent_seqs=n_frequent_sequences,
                                               n_cluster_centers=n_cluster_centers,
                                               seq_length=seq_length,
                                               seq_lookahead=2,
                                               device=device)

    frequent_sequences = torch.tensor([[[0, 1, 2, 3], [0, 2, 3, 2], [1, 3, 2, 0], [3, 2, 3, 2]],
                                       [[1, 3, 2, 1], [0, 1, 0, 3], [3, 2, 0, 1], [1, 3, 2, 3]]], dtype=torch.int64,
                                      device=device)

    pred_outputs1 = torch.zeros(flock_size, seq_length, n_cluster_centers, dtype=float_dtype, device=device)
    pred_outputs2 = torch.zeros(flock_size, seq_length, n_cluster_centers, dtype=float_dtype, device=device)

    sequence_likelihoods1 = torch.tensor([[3, 1, 2, 2], [4, 1, 2, 4]], dtype=float_dtype, device=device)
    sequence_likelihoods2 = torch.tensor([[2, 1, 1, 1], [2, 3, 1, 2]], dtype=float_dtype, device=device)

    expected_pred_outputs1 = torch.tensor([[[0.5, 0.25, 0, 0.25],
                                            [0, 0.375, 0.375, 0.25],
                                            [0, 0, 0.625, 0.375],
                                            [0.25, 0, 0.375, 0.375]],
                                           [[0.0909, 0.7273, 0, 0.1818],
                                            [0, 0.0909, 0.1818, 0.7272],
                                            [0.2727, 0, 0.7273, 0],
                                            [0.0000, 0.5455, 0.0000, 0.4545]]], dtype=float_dtype, device=device)

    expected_pred_outputs2 = torch.tensor([[[0.6, 0.2, 0, 0.2],
                                            [0, 0.4, 0.4, 0.2],
                                            [0, 0, 0.6, 0.4],
                                            [0.2, 0, 0.4, 0.4]],
                                           [[0.375, 0.5, 0, 0.125],
                                            [0, 0.375, 0.125, 0.5],
                                            [0.5, 0, 0.5, 0],
                                            [0, 0.3750, 0, 0.6250]]], dtype=float_dtype, device=device)

    # endregion

    process1._compute_predicted_clusters(frequent_sequences, sequence_likelihoods1, pred_outputs1)
    process2._compute_predicted_clusters(frequent_sequences, sequence_likelihoods2, pred_outputs2)

    assert same(expected_pred_outputs1, pred_outputs1, eps=1e-3)
    assert same(expected_pred_outputs2, pred_outputs2, eps=1e-3)


# @pytest.mark.skip("Method _compute_exploration_likelihoods() is not used anymore.")
# def test_compute_exploration_likelihoods():
#     device = 'cuda'
#     float_dtype = get_float(device)
#     flock_size = 2
#     n_frequent_seqs = 3
#
#     frequent_exploration_attempts = torch.tensor([[[2], [3], [0]],
#                                                   [[2], [0], [5]]], dtype=float_dtype, device=device)
#
#     passive_seq_likelihoods = torch.tensor([[0.2, 0.3, 0], [3, 4, 1]], dtype=float_dtype, device=device)
#
#     seq_likelihoods_exploration = torch.zeros((flock_size, n_frequent_seqs), dtype=float_dtype, device=device)
#
#     TPFlockForwardAndBackward._compute_exploration_likelihoods(frequent_exploration_attempts, passive_seq_likelihoods,
#                                                                seq_likelihoods_exploration)
#
#     # Difference in likelihoods between f6 amd f32 is ~2, so prepare different expected values. Also, epsilon is smaller.
#     if float_dtype == torch.float16:
#         expected_seq_likelihoods_exploration = torch.tensor([[0.1, 0.1, 0], [1.5, 3998, 0.2]], dtype=float_dtype,
#                                                             device=device)
#     else:
#         expected_seq_likelihoods_exploration = torch.tensor([[0.1, 0.1, 0], [1.5, 40000, 0.2]], dtype=float_dtype,
#                                                             device=device)
#
#     assert same(expected_seq_likelihoods_exploration, seq_likelihoods_exploration, eps=1e-1)


@pytest.mark.flaky(reruns=3)
def test_exploration():
    device = 'cuda'

    flock_size = 150
    n_frequent_sequences = 4
    n_cluster_centers = 4
    seq_length = 4

    exploration_probability = 1

    process = create_tp_flock_forward_process(flock_size=flock_size,
                                              n_frequent_seqs=n_frequent_sequences,
                                              n_cluster_centers=n_cluster_centers,
                                              seq_length=seq_length,
                                              cluster_exploration_prob=exploration_probability,
                                              exploration_probability=exploration_probability,
                                              device=device)

    process._action_outputs.fill_(-2)
    process.exploring.fill_(1)
    process.exploration_random_numbers.fill_(0.99)

    process._exploration_cluster(process.exploration_random_numbers, process._action_outputs, process._action_rewards)

    exploration_actions = process._action_outputs[process.exploring.squeeze() == 1]
    non_exploration_actions = process._action_outputs[process.exploring.squeeze() == 0]

    # When not exploring, actions should not change
    assert (non_exploration_actions == -2).all()

    # Should only be a single value of EXPLORATION_REWARD, all other values should be zero
    vals, indices = process._action_rewards.max(dim=1)
    assert (vals == EXPLORATION_REWARD).all()
    exp_copy = exploration_actions.clone()
    exp_copy[:, indices] = 0
    assert (exp_copy == 0).all()

    # should be random, which is difficult to test, so at least test that the actions are tried with similar
    # probabilities
    no_attempts = exploration_actions.sum(dim=0)
    assert no_attempts.min() * 2 > no_attempts.max()
