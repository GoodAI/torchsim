import pytest
import torch

from torchsim.core import get_float, FLOAT_NAN
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.expert_params import DEFAULT_CONTEXT_PRIOR, DEFAULT_EXPLORATION_ATTEMPTS_PRIOR
from torchsim.core.models.temporal_pooler import TPFlockLearning, TPFlockBuffer
from torchsim.core.utils.tensor_utils import same

from torchsim.core.kernels import check_cuda_errors


def create_tp_buffer(flock_size=2,
                     buffer_size=5,
                     n_cluster_centers=3,
                     n_frequent_seqs=3,
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


def create_tp_flock_learn_process(buffer=None,
                                  all_encountered_seqs=None,
                                  all_encountered_seq_occurrences=None,
                                  all_encountered_context_occurrences=None,
                                  all_encountered_exploration_attempts=None,
                                  all_encountered_exploration_success_rates=None,
                                  frequent_seqs=None,
                                  frequent_seq_occurrences=None,
                                  frequent_context_likelihoods=None,
                                  frequent_exploration_attempts=None,
                                  frequent_exploration_results=None,
                                  execution_counter_learning=None,
                                  all_encountered_rewards_punishments=None,
                                  frequent_rewards_punishments=None,
                                  n_providers=1,
                                  flock_size=2,
                                  max_encountered_seqs=7,
                                  n_frequent_seqs=3,
                                  seq_length=3,
                                  n_cluster_centers=3,
                                  batch_size=5,
                                  forgetting_limit=3,
                                  context_size=4,
                                  context_prior=DEFAULT_CONTEXT_PRIOR,
                                  do_subflocking=True,
                                  seq_lookahead=1,
                                  exploration_attempts_prior=DEFAULT_EXPLORATION_ATTEMPTS_PRIOR,
                                  n_subbatches=1,
                                  max_new_seqs=3,
                                  device='cuda'):
    float_dtype = get_float(device)
    seq_lookbehind = seq_length - seq_lookahead

    all_indices = torch.arange(end=flock_size, device=device).unsqueeze(dim=1)

    if buffer is None:
        buffer = create_tp_buffer(flock_size=flock_size, device=device)

    if frequent_seqs is None:
        frequent_seqs = torch.full((flock_size, n_frequent_seqs, seq_length), fill_value=-1.,
                                   dtype=torch.int64, device=device)

    if frequent_seq_occurrences is None:
        frequent_seq_occurrences = torch.full((flock_size, n_frequent_seqs), fill_value=-1., dtype=float_dtype,
                                              device=device)

    if frequent_context_likelihoods is None:
        frequent_context_likelihoods = torch.full((flock_size, n_frequent_seqs, seq_length, n_providers, context_size),
                                                  fill_value=-1., dtype=float_dtype, device=device)

    if frequent_exploration_attempts is None:
        frequent_exploration_attempts = torch.full((flock_size, n_frequent_seqs, seq_lookahead),
                                                   fill_value=-1., dtype=float_dtype, device=device)

    if frequent_exploration_results is None:
        frequent_exploration_results = torch.full((flock_size, n_frequent_seqs, seq_lookahead, n_cluster_centers),
                                                  fill_value=-1., dtype=float_dtype, device=device)

    if all_encountered_exploration_attempts is None:
        all_encountered_exploration_attempts = torch.full(
            (flock_size, max_encountered_seqs, seq_lookahead),
            fill_value=-1., dtype=float_dtype, device=device)

    if all_encountered_exploration_success_rates is None:
        all_encountered_exploration_success_rates = torch.full(
            (flock_size, max_encountered_seqs, seq_lookahead, n_cluster_centers),
            fill_value=-1., dtype=float_dtype, device=device)

    if execution_counter_learning is None:
        execution_counter_learning = torch.zeros((flock_size, 1), device=device, dtype=float_dtype)

    if all_encountered_seqs is None:
        all_encountered_seqs = torch.full((flock_size, max_encountered_seqs, seq_length), fill_value=-1,
                                          dtype=float_dtype, device=device)

    if all_encountered_seq_occurrences is None:
        all_encountered_seq_occurrences = torch.zeros((flock_size, max_encountered_seqs), dtype=float_dtype,
                                                      device=device)

    if all_encountered_context_occurrences is None:
        all_encountered_context_occurrences = torch.full(
            (flock_size, max_encountered_seqs, seq_length, n_providers, context_size),
            fill_value=-1, dtype=float_dtype, device=device)

    if all_encountered_rewards_punishments is None:
        all_encountered_rewards_punishments = torch.zeros((flock_size, max_encountered_seqs, seq_lookahead, 2),
                                                          dtype=float_dtype, device=device)

    if frequent_rewards_punishments is None:
        frequent_rewards_punishments = torch.zeros((flock_size, n_frequent_seqs, seq_lookahead, 2), dtype=float_dtype,
                                                   device=device)

    return TPFlockLearning(all_indices,
                           do_subflocking,
                           buffer,
                           all_encountered_seqs,
                           all_encountered_seq_occurrences,
                           all_encountered_context_occurrences,
                           all_encountered_rewards_punishments,
                           all_encountered_exploration_attempts,
                           all_encountered_exploration_success_rates,
                           frequent_seqs,
                           frequent_seq_occurrences,
                           frequent_context_likelihoods,
                           frequent_rewards_punishments,
                           frequent_exploration_attempts,
                           frequent_exploration_results,
                           execution_counter_learning,
                           max_encountered_seqs,
                           max_new_seqs,
                           n_frequent_seqs,
                           seq_length,
                           seq_lookahead,
                           seq_lookbehind,
                           n_cluster_centers,
                           batch_size,
                           forgetting_limit,
                           context_size,
                           context_prior,
                           exploration_attempts_prior,
                           n_subbatches,
                           n_providers,
                           device)


def test_learn():
    device = 'cuda'
    float_dtype = get_float(device)
    flock_size = 2
    n_frequent_seqs = 3
    context_size = 4
    seq_length = 3
    max_encountered_seqs = 7
    n_providers = 1

    # region Setup
    creator = AllocatingCreator(device)
    buffer = TPFlockBuffer(creator=creator, flock_size=flock_size, buffer_size=6, n_cluster_centers=3,
                           n_frequent_seqs=n_frequent_seqs, context_size=context_size, n_providers=n_providers)

    buffer.clusters.stored_data = torch.tensor([[[0, 1, 0],
                                                 [1, 0, 0],
                                                 [0, 1, 0],
                                                 [1, 0, 0],
                                                 [1, 0, 0],
                                                 [-1, -1, -1]],
                                                [[0, 1, 0],
                                                 [0, 0, 1],
                                                 [1, 0, 0],
                                                 [0, 0, 1],
                                                 [1, 0, 0],
                                                 [-1, -1, -1]]], dtype=float_dtype, device=device)

    buffer.contexts.stored_data = torch.tensor([[[[1, 1, 1, 1]],
                                                 [[0, 0, 0, 0]],
                                                 [[0, 1, 0, 1]],
                                                 [[1, 0.5, 0.5, 0.2]],
                                                 [[1, 0, 0, 1]],
                                                 [[-1, -1, -1, -1]]],
                                                [[[1, 1, 0, 1]],
                                                 [[1, 1, 0, 0.2]],
                                                 [[0.9, 0, 0, 1]],
                                                 [[1, 0, 0, 1]],
                                                 [[0, 1, 0, 0]],
                                                 [[-1, -1, -1, -1]]]], dtype=float_dtype, device=device)

    buffer.current_ptr = torch.tensor([4, 4], dtype=torch.int64, device=device)

    all_encountered_seqs = torch.tensor([[[1, 0, 1],
                                          [2, 1, 0],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1]],
                                         [[1, 0, 1],
                                          [0, 1, 2],
                                          [1, 2, 0],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1]]], dtype=torch.int64, device=device)

    all_encountered_seq_occurrences = torch.tensor([[0.55, 0.4, 0, 0, 0, 0, 0],
                                                    [0.5, 0.5, 0.5, 0, 0, 0, 0]], dtype=float_dtype, device=device)

    all_encountered_context_occurrences = torch.tensor([[[[[0, 0.2, 0, 0]],
                                                          [[0, 0.1, 0.2, 0.3]],
                                                          [[0.2, 0.1, 0, 0]]],
                                                         [[[0.2, 0.2, 0.2, 0.2]],
                                                          [[0.2, 0.2, 0.2, 0.2]],
                                                          [[0.1, 0.1, 0.1, 0.1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]]],
                                                        [[[[0, 0, 0, 0.1]],
                                                          [[0, 0, 0, 0]],
                                                          [[0, 0, 0, 0]]],
                                                         [[[0, 0, 0, 0.2]],
                                                          [[0, 0, 0, 0]],
                                                          [[0, 0, 0, 0]]],
                                                         [[[0, 0, 0, 0.3]],
                                                          [[0, 0, 0, 0]],
                                                          [[0, 0, 0, 0]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]]]], dtype=float_dtype, device=device)

    # just pre-fill with some values to check that they were rewritten
    frequent_seqs = torch.full((flock_size, n_frequent_seqs, seq_length), fill_value=-1.,
                               dtype=torch.int64, device=device)

    frequent_seq_occurrences = torch.full((flock_size, n_frequent_seqs), fill_value=-1., dtype=float_dtype,
                                          device=device)

    frequent_pos_context_probs = torch.full((flock_size, n_frequent_seqs, seq_length, n_providers, context_size),
                                            fill_value=-1., dtype=float_dtype, device=device)
    # endregion

    # region ExpectedOutputs
    expected_all_encountered_seqs = torch.tensor([[[1, 0, 1],
                                                   [0, 1, 0],
                                                   [2, 1, 0],
                                                   [-1, -1, -1],
                                                   [-1, -1, -1],
                                                   [-1, -1, -1],
                                                   [-1, -1, -1]],
                                                  [[1, 2, 0],
                                                   [0, 2, 0],
                                                   [2, 0, 2],
                                                   [0, 1, 2],
                                                   [1, 0, 1],
                                                   [-1, -1, -1],
                                                   [-1, -1, -1]]], dtype=torch.int64, device=device)

    expected_all_encountered_seq_occurrences = torch.tensor([[1.55, 1., 0.4, 0., 0., 0., 0.],
                                                             [1., 0.6667, 0.6667, 0.3333, 0.3333, 0., 0.]],
                                                            dtype=float_dtype,
                                                            device=device)

    expected_frequent_seqs = torch.tensor([[[1, 0, 1],
                                            [0, 1, 0],
                                            [2, 1, 0]],
                                           [[1, 2, 0],
                                            [0, 2, 0],
                                            [2, 0, 2]]], dtype=torch.int64, device=device)
    expected_frequent_seq_occurrences = torch.tensor([[1.55, 1., 0.4], [1., 0.6667, 0.6667]], dtype=float_dtype,

                                                     device=device)

    expected_all_encountered_context_occurrences = torch.tensor([[[[[1, 1.2, 1, 1]],
                                                                   [[0, 0.1, 0.2, 0.3]],
                                                                   [[0.2, 1.1, 0, 1]]],
                                                                  [[[0.5, 0.5, 0.5, 0.5]],
                                                                   [[0.5, 0.5, 0.5, 0.5]],
                                                                   [[0.5, 0.5, 0.5, 0.5]]],
                                                                  [[[0.2, 0.2, 0.2, 0.2]],
                                                                   [[0.2, 0.2, 0.2, 0.2]],
                                                                   [[0.1, 0.1, 0.1, 0.1]]],
                                                                  [[[0, 0, 0, 0]],
                                                                   [[0, 0, 0, 0]],
                                                                   [[0, 0, 0, 0]]],
                                                                  [[[0, 0, 0, 0]],
                                                                   [[0, 0, 0, 0]],
                                                                   [[0, 0, 0, 0]]],
                                                                  [[[-1, -1, -1, -1]],
                                                                   [[-1, -1, -1, -1]],
                                                                   [[-1, -1, -1, -1]]],
                                                                  [[[-1, -1, -1, -1]],
                                                                   [[-1, -1, -1, -1]],
                                                                   [[-1, -1, -1, -1]]]],
                                                                 [[[[0.6667, 0.6667, 0, 0.8667]],
                                                                   [[0.6667, 0.6667, 0, 0.13333]],
                                                                   [[0.6, 0, 0, 0.6667]]],
                                                                  [[[0.3333, 0.3333, 0.3333, 0.3333]],
                                                                   [[0.3333, 0.3333, 0.3333, 0.3333]],
                                                                   [[0.3333, 0.3333, 0.3333, 0.3333]]],
                                                                  [[[0.3333, 0.3333, 0.3333, 0.3333]],
                                                                   [[0.3333, 0.3333, 0.3333, 0.3333]],
                                                                   [[0.3333, 0.3333, 0.3333, 0.3333]]],
                                                                  [[[0, 0, 0, 0.1333]],
                                                                   [[0, 0, 0, 0]],
                                                                   [[0, 0, 0, 0]]],
                                                                  [[[0, 0, 0, 0.0667]],
                                                                   [[0, 0, 0, 0]],
                                                                   [[0, 0, 0, 0]]],
                                                                  [[[-0.6667, -0.6667, -0.6667, -0.6667]],
                                                                   [[-0.6667, -0.6667, -0.6667, -0.6667]],
                                                                   [[-0.6667, -0.6667, -0.6667, -0.6667]]],
                                                                  [[[0, 0, 0, 0]],
                                                                   [[0, 0, 0, 0]],
                                                                   [[0, 0, 0, 0]]]]], dtype=float_dtype,
                                                                device=device)

    expected_frequent_context_likelihoods = torch.tensor([[[[[0.5195, 0.5368, 0.5195, 0.5195]],
                                                            [[0.4329, 0.4416, 0.4502, 0.4589]],
                                                            [[0.4502, 0.5281, 0.4329, 0.5195]]],

                                                           [[[0.5000, 0.5000, 0.5000, 0.5000]],
                                                            [[0.5000, 0.5000, 0.5000, 0.5000]],
                                                            [[0.5000, 0.5000, 0.5000, 0.5000]]],

                                                           [[[0.5000, 0.5000, 0.5000, 0.5000]],
                                                            [[0.5000, 0.5000, 0.5000, 0.5000]],
                                                            [[0.4904, 0.4904, 0.4904, 0.4904]]]],

                                                          [[[[0.5152, 0.5152, 0.4545, 0.5333]],
                                                            [[0.5152, 0.5152, 0.4545, 0.4667]],
                                                            [[0.5091, 0.4545, 0.4545, 0.5152]]],

                                                           [[[0.5000, 0.5000, 0.5000, 0.5000]],
                                                            [[0.5000, 0.5000, 0.5000, 0.5000]],
                                                            [[0.5000, 0.5000, 0.5000, 0.5000]]],

                                                           [[[0.5000, 0.5000, 0.5000, 0.5000]],
                                                            [[0.5000, 0.5000, 0.5000, 0.5000]],
                                                            [[0.5000, 0.5000, 0.5000, 0.5000]]]]],
                                                         dtype=float_dtype, device=device)

    # endregion

    # region ExpectedBookkeeping
    expected_encountered_batch_occurrences = torch.tensor([[1, 0, 0, 0, 0, 0, 0],
                                                           [0, 0, 1, 0, 0, 0, 0]], dtype=float_dtype, device=device)
    # expected_most_probable_batch_seq_probs = torch.tensor([[0, 1, 0], [0, 1, 1]], dtype=float_dtype,
    #                                                       device=device)
    expected_most_probable_batch_seq_probs = torch.tensor([[1, 0, 0], [1, 0, 1]], dtype=float_dtype,
                                                          device=device)
    # expected_most_probable_batch_seqs = torch.tensor([[[-1, -1, -1],
    #                                                    [0, 1, 0],
    #                                                    [-1, -1, -1]],
    #                                                   [[-1, -1, -1],
    #                                                    [2, 0, 2],
    #                                                    [0, 2, 0]]], dtype=torch.int64, device=device)
    expected_most_probable_batch_seqs = torch.tensor([[[0, 1, 0],
                                                       [-1, -1, -1],
                                                       [-1, -1, -1]],
                                                      [[0, 2, 0],
                                                       [-1, -1, -1],
                                                       [2, 0, 2]]], dtype=torch.int64, device=device)
    expected_newly_encountered_seqs_counts = torch.tensor([[1, 0, 0], [1, 0, 1]], dtype=float_dtype, device=device)
    expected_newly_encountered_seqs_indicator = torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.int64,
                                                             device=device)
    expected_total_encountered_occurrences = torch.tensor([2.95, 3], dtype=float_dtype, device=device)

    # endregion

    process = create_tp_flock_learn_process(buffer=buffer,
                                            all_encountered_seqs=all_encountered_seqs,
                                            all_encountered_seq_occurrences=all_encountered_seq_occurrences,
                                            all_encountered_context_occurrences=all_encountered_context_occurrences,
                                            frequent_seqs=frequent_seqs,
                                            frequent_seq_occurrences=frequent_seq_occurrences,
                                            frequent_context_likelihoods=frequent_pos_context_probs,
                                            flock_size=flock_size,
                                            context_size=context_size,
                                            max_encountered_seqs=max_encountered_seqs,
                                            n_frequent_seqs=n_frequent_seqs,
                                            seq_length=seq_length)

    process.run_and_integrate()

    # Assert integrity of bookkeeping tensors
    assert same(expected_encountered_batch_occurrences, process.encountered_batch_seq_occurrences)
    assert same(expected_most_probable_batch_seq_probs, process.most_probable_batch_seq_probs)
    assert same(expected_most_probable_batch_seqs, process.most_probable_batch_seqs)
    assert same(expected_newly_encountered_seqs_counts, process.newly_encountered_seqs_counts)
    assert same(expected_newly_encountered_seqs_indicator, process.newly_encountered_seqs_indicator)
    assert same(expected_total_encountered_occurrences, process.total_encountered_occurrences, eps=0.0001)

    # Assert the integrity of the return values
    assert same(expected_all_encountered_seqs, all_encountered_seqs)
    assert same(expected_all_encountered_seq_occurrences, all_encountered_seq_occurrences, eps=0.0001)
    assert same(expected_frequent_seqs, frequent_seqs)
    assert same(expected_frequent_seq_occurrences, frequent_seq_occurrences, eps=0.0001)
    assert same(expected_all_encountered_context_occurrences, all_encountered_context_occurrences, eps=1e-3)
    assert same(expected_frequent_context_likelihoods, frequent_pos_context_probs, eps=1e-3)


def test_learn_from_batch():
    device = 'cuda'
    float_dtype = get_float(device)
    seq_lookahead = 2
    seq_length = 3
    n_providers = 1
    n_cluster_centers = 3

    # region Setup
    cluster_batch = torch.tensor([[[0, 1, 0],
                                   [1, 0, 0],
                                   [0, 1, 0],
                                   [1, 0, 0],
                                   [1, 0, 0]],
                                  [[0, 1, 0],
                                   [0, 0, 1],
                                   [1, 0, 0],
                                   [0, 0, 1],
                                   [1, 0, 0]]], dtype=float_dtype, device=device)

    context_batch = torch.tensor([[[[1, 0, 1, 1]],
                                   [[0.2, 0.8, 0.33, 1]],
                                   [[0.6, 0.99, 1, 0]],
                                   [[0, 1, 1, 0]],
                                   [[0, 1, 1, 1]]],
                                  [[[1, 1, 1, 1]],
                                   [[0, 0, 0, 0]],
                                   [[1, 1, 0, 0]],
                                   [[0, 0, 1, 1]],
                                   [[0, 1, 0, 1]]]], dtype=float_dtype, device=device)

    rewards_punishments_batch = torch.tensor([[[0.5, 0],
                                               [1.2, 0],
                                               [0, 0.53],
                                               [1.2, 0],
                                               [1.1, 0]],
                                              [[0, 1.9],
                                               [0, 0],
                                               [1.2, 0],
                                               [0, 0],
                                               [1.3, 0]]], dtype=float_dtype, device=device)

    exploring_batch = torch.tensor([[[1],
                                     [0],
                                     [1],
                                     [0],
                                     [1]],
                                    [[0],
                                     [1],
                                     [1],
                                     [1],
                                     [0]]], dtype=float_dtype, device=device)

    actions_batch = torch.tensor([[[0.5, 0.5, 0],
                                   [0, 0.2, 0.8],
                                   [1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]],
                                  [[0, 0.5, 0.5],
                                   [0, 0, 1],
                                   [0, 1, 0],
                                   [0, 0.5, 0.5],
                                   [0.2, 0.4, 0.4]]], dtype=float_dtype, device=device)

    all_encountered_seqs = torch.tensor([[[1, 0, 1],
                                          [2, 1, 0],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1]],
                                         [[1, 0, 1],
                                          [0, 1, 2],
                                          [1, 2, 0],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1]]], dtype=torch.int64, device=device)

    all_encountered_seq_occurrences = torch.tensor([[0.5, 0.5, 0, 0, 0, 0, 0],
                                                    [0.5, 0.5, 0.5, 0, 0, 0, 0]], dtype=float_dtype, device=device)

    all_encountered_context_occurrences = torch.tensor([[[[[5, 8.2, 0, 13]],
                                                          [[0, 1, 2, 3]],
                                                          [[5.2, 2, 0, 0]]],
                                                         [[[2, 2, 2, 2]],
                                                          [[2, 2, 2, 2]],
                                                          [[1, 1, 1, 1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]]],
                                                        [[[[0, 0, 1, 0.1]],
                                                          [[1, 1, 1, 1]],
                                                          [[1, 2, 1, 5]]],
                                                         [[[0, 0, 1, 0.1]],
                                                          [[1, 1, 1, 1]],
                                                          [[1, 2, 1, 5]]],
                                                         [[[0, 0, 1, 0.1]],
                                                          [[1, 1, 1, 1]],
                                                          [[1, 2, 1, 5]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]],
                                                         [[[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]],
                                                          [[-1, -1, -1, -1]]]]], dtype=float_dtype, device=device)

    all_encountered_rewards_punishments = torch.tensor([[[[0.32, 0.36],
                                                          [1, 0.0]],
                                                         [[0., 1],
                                                          [0.4, 0.4]],
                                                         [[0, 0],
                                                          [0, 0]],
                                                         [[0, 0],
                                                          [0, 0]],
                                                         [[0, 0],
                                                          [0, 0]],
                                                         [[0, 0],
                                                          [0, 0]],
                                                         [[0, 0],
                                                          [0, 0]]],
                                                        [[[0.0, 1.0],
                                                          [0.1, 0.0]],
                                                         [[0.3, 0.3],
                                                          [0.4, 0.4]],
                                                         [[0.0, 0.0],
                                                          [1, 0.0]],
                                                         [[0, 0],
                                                          [0, 0]],
                                                         [[0, 0],
                                                          [0, 0]],
                                                         [[0, 0],
                                                          [0, 0]],
                                                         [[0, 0],
                                                          [0, 0]]]], dtype=float_dtype, device=device)

    prior = DEFAULT_EXPLORATION_ATTEMPTS_PRIOR
    all_encountered_exploration_attempts = torch.tensor([[[0.1, 0.2],
                                                          [0.3, 0.4],
                                                          [prior, prior],
                                                          [prior, prior],
                                                          [prior, prior],
                                                          [prior, prior],
                                                          [prior, prior]],
                                                         [[0.1, 0.2],
                                                          [0.3, 0.4],
                                                          [0.5, 0.6],
                                                          [prior, prior],
                                                          [prior, prior],
                                                          [prior, prior],
                                                          [prior, prior]]], dtype=float_dtype, device=device)

    prior = 0
    all_encountered_exploration_results = torch.tensor([[[[0.32, 0.36, 0.32],
                                                          [1, 0.0, 0.0]],
                                                         [[0., 1, 0.3],
                                                          [0.4, 0.4, 0.4]],
                                                         [[prior, 1, prior],
                                                          [1, prior, prior]],
                                                         [[prior, prior, prior],
                                                          [prior, prior, prior]],
                                                         [[prior, prior, prior],
                                                          [prior, prior, prior]],
                                                         [[prior, prior, prior],
                                                          [prior, prior, prior]],
                                                         [[prior, prior, prior],
                                                          [prior, prior, prior]]],
                                                        [[[0.0, 1.0, 0.0],
                                                          [0.1, 0.0, 0.0]],
                                                         [[0.3, 0.3, 0.3],
                                                          [0.4, 0.4, 0.4]],
                                                         [[0.0, 0.0, 1],
                                                          [1, 0.0, 0.0]],
                                                         [[prior, 1, prior],
                                                          [prior, prior, 1]],
                                                         [[1, prior, prior],
                                                          [prior, 1, prior]],
                                                         [[prior, prior, prior],
                                                          [prior, prior, prior]],
                                                         [[prior, prior, prior],
                                                          [prior, prior, prior]]]],
                                                       dtype=float_dtype, device=device)
    # endregion

    # region ExpectedOutputs
    expected_all_encountered_seqs = torch.tensor([[[1, 0, 1],
                                                   [0, 1, 0],
                                                   [2, 1, 0],
                                                   [-1, -1, -1],
                                                   [-1, -1, -1],
                                                   [-1, -1, -1],
                                                   [-1, -1, -1]],
                                                  [[1, 2, 0],
                                                   [0, 2, 0],
                                                   [2, 0, 2],
                                                   [0, 1, 2],
                                                   [1, 0, 1],
                                                   [-1, -1, -1],
                                                   [-1, -1, -1]]], dtype=torch.int64, device=device)

    expected_all_encountered_occurrences = torch.tensor([[1.5, 1., 0.5, 0., 0., 0., 0.],
                                                         [1.5, 1., 1., 0.5, 0.5, 0., 0.]], dtype=float_dtype,
                                                        device=device)

    expected_all_encountered_context_occurrences = torch.tensor([[[[[6, 8.2, 1, 14]],
                                                                   [[0.2, 1.8, 2.33, 4]],
                                                                   [[5.8, 2.99, 1, 0]]],
                                                                  [[[0.5, 0.5, 0.5, 0.5]],
                                                                   [[0.5, 0.5, 0.5, 0.5]],
                                                                   [[0.5, 0.5, 0.5, 0.5]]],
                                                                  [[[2, 2, 2, 2]],
                                                                   [[2, 2, 2, 2]],
                                                                   [[1, 1, 1, 1]]],
                                                                  [[[0, 0, 0, 0]],
                                                                   [[0, 0, 0, 0]],
                                                                   [[0, 0, 0, 0]]],
                                                                  [[[0, 0, 0, 0]],
                                                                   [[0, 0, 0, 0]],
                                                                   [[0, 0, 0, 0]]],
                                                                  [[[-1, -1, -1, -1]],
                                                                   [[-1, -1, -1, -1]],
                                                                   [[-1, -1, -1, -1]]],
                                                                  [[[-1, -1, -1, -1]],
                                                                   [[-1, -1, -1, -1]],
                                                                   [[-1, -1, -1, -1]]]],
                                                                 [[[[1, 1, 2, 1.1]],
                                                                   [[1, 1, 1, 1]],
                                                                   [[2, 3, 1, 5]]],
                                                                  [[[0.5, 0.5, 0.5, 0.5]],
                                                                   [[0.5, 0.5, 0.5, 0.5]],
                                                                   [[0.5, 0.5, 0.5, 0.5]]],
                                                                  [[[0.5, 0.5, 0.5, 0.5]],
                                                                   [[0.5, 0.5, 0.5, 0.5]],
                                                                   [[0.5, 0.5, 0.5, 0.5]]],
                                                                  [[[0, 0, 1, 0.1]],
                                                                   [[1, 1, 1, 1]],
                                                                   [[1, 2, 1, 5]]],
                                                                  [[[0, 0, 1, 0.1]],
                                                                   [[1, 1, 1, 1]],
                                                                   [[1, 2, 1, 5]]],
                                                                  [[[-1, -1, -1, -1]],
                                                                   [[-1, -1, -1, -1]],
                                                                   [[-1, -1, -1, -1]]],
                                                                  [[[0, 0, 0, 0]],
                                                                   [[0, 0, 0, 0]],
                                                                   [[0, 0, 0, 0]]]]], dtype=float_dtype,
                                                                device=device)

    expected_all_encountered_rewards_punishments = torch.tensor([[[[1.52, 0.36],
                                                          [1, 0.53]],
                                                         [[0., 0],
                                                          [0.0, 0.0]],
                                                         [[0, 1],
                                                          [0.4, 0.4]],
                                                         [[0, 0],
                                                          [0, 0]],
                                                         [[0, 0],
                                                          [0, 0]],
                                                         [[0, 0],
                                                          [0, 0]],
                                                         [[0, 0],
                                                          [0, 0]]],
                                                        [[[0.0, 0.0],
                                                          [2.2, 0.0]],
                                                         [[0.0, 0.0],
                                                          [0.0, 0.0]],
                                                         [[0.0, 0.0],
                                                          [0.0, 0.0]],
                                                         [[0.3, 0.3],
                                                          [0.4, 0.4]],
                                                         [[0.0, 1],
                                                          [0.1, 0]],
                                                         [[0, 0],
                                                          [0, 0]],
                                                         [[0, 0],
                                                          [0, 0]]]], dtype=float_dtype, device=device)

    prior = DEFAULT_EXPLORATION_ATTEMPTS_PRIOR
    expected_all_encountered_exploration_attempts = torch.tensor([[[1.6, 0.2],
                                                                   [prior, prior],
                                                                   [0.3, 0.4],
                                                                   [prior, prior],
                                                                   [prior, prior],
                                                                   [prior, prior],
                                                                   [prior, prior]],

                                                                  [[0.5, 0.6],
                                                                   [prior, prior],
                                                                   [prior, prior],
                                                                   [1.3, 0.4],
                                                                   [0.1, 0.2],
                                                                   [prior, prior],
                                                                   [prior, prior]]],
                                                                 dtype=float_dtype, device=device)

    prior = 0
    expected_all_encountered_exploration_results = torch.tensor([[[[0.957, 0.022, 0.020],
                                                                   [1., 0., 0.]],

                                                                  [[prior, 1, prior],
                                                                   [1, prior, prior]],

                                                                  [[0., 1., 0.3],
                                                                   [0.4, 0.4, 0.4]],

                                                                  [[prior, prior, prior],
                                                                   [prior, prior, prior]],

                                                                  [[prior, prior, prior],
                                                                   [prior, prior, prior]],

                                                                  [[prior, 1, prior],
                                                                   [1, prior, prior]],

                                                                  [[prior, prior, prior],
                                                                   [prior, prior, prior]]],

                                                                 [[[0., 0., 1.],
                                                                   [1., 0., 0.]],

                                                                  [[prior, prior, 1],
                                                                   [1, prior, prior]],

                                                                  [[1, prior, prior],
                                                                   [prior, prior, 1]],

                                                                  [[0.069, 0.069, 0.83],
                                                                   [0.4, 0.4, 0.4]],

                                                                  [[0., 1., 0.],
                                                                   [0.1, 0., 0.]],

                                                                  [[prior, 1, prior],
                                                                   [prior, prior, 1]],

                                                                  [[prior, prior, prior],
                                                                   [prior, prior, prior]]]],
                                                                dtype=float_dtype, device=device)
    # endregion

    # region ExpectedBookkeeping
    expected_encountered_batch_occurrences = torch.tensor([[1, 0, 0, 0, 0, 0, 0],
                                                           [0, 0, 1, 0, 0, 0, 0]], dtype=float_dtype, device=device)
    expected_most_probable_batch_seq_probs = torch.tensor([[1, 0, 0], [1, 0, 1]], dtype=float_dtype,
                                                          device=device)
    expected_most_probable_batch_seqs = torch.tensor([[[0, 1, 0],
                                                       [-1, -1, -1],
                                                       [-1, -1, -1]],
                                                      [[0, 2, 0],
                                                       [-1, -1, -1],
                                                       [2, 0, 2]]], dtype=torch.int64, device=device)
    expected_newly_encountered_seqs_counts = torch.tensor([[1, 0, 0], [1, 0, 1]], dtype=float_dtype, device=device)
    expected_newly_encountered_seqs_indicator = torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.int64,
                                                             device=device)

    expected_encountered_batch_exploration_attempts = torch.tensor([[[1.5, 0],
                                                                     [0, 0],
                                                                     [0, 0],
                                                                     [0, 0],
                                                                     [0, 0],
                                                                     [0, 0],
                                                                     [0, 0]],
                                                                    [[0, 0],
                                                                     [1, 0],
                                                                     [0, 0],
                                                                     [0, 0],
                                                                     [0, 0],
                                                                     [0, 0],
                                                                     [0, 0]]],
                                                                   dtype=float_dtype, device=device)

    expected_encountered_batch_exploration_results = torch.tensor([[[[1.5, 0, 0], [0, 0, 0]],
                                                                    [[0, 0, 0], [0, 0, 0]],
                                                                    [[0, 0, 0], [0, 0, 0]],
                                                                    [[0, 0, 0], [0, 0, 0]],
                                                                    [[0, 0, 0], [0, 0, 0]],
                                                                    [[0, 0, 0], [0, 0, 0]],
                                                                    [[0, 0, 0], [0, 0, 0]]],
                                                                   [[[0, 0, 0], [0, 0, 0]],
                                                                    [[0, 0, 1], [0, 0, 0]],
                                                                    [[0, 0, 0], [0, 0, 0]],
                                                                    [[0, 0, 0], [0, 0, 0]],
                                                                    [[0, 0, 0], [0, 0, 0]],
                                                                    [[0, 0, 0], [0, 0, 0]],
                                                                    [[0, 0, 0], [0, 0, 0]]]],
                                                                  dtype=float_dtype, device=device)

    # endregion

    process = create_tp_flock_learn_process(None,
                                            all_encountered_seqs,
                                            all_encountered_seq_occurrences,
                                            all_encountered_context_occurrences,
                                            seq_lookahead=seq_lookahead,
                                            seq_length=seq_length,
                                            n_providers=n_providers,
                                            n_cluster_centers=n_cluster_centers,
                                            device=device)

    cluster_subbatch = cluster_batch.unsqueeze(1)
    context_subbatch = context_batch.unsqueeze(1)
    exploring_subbatch = exploring_batch.unsqueeze(1)
    actions_subbatch = actions_batch.unsqueeze(1)
    rewards_punishments_subbatch = rewards_punishments_batch.unsqueeze(1)

    process._learn_from_batch(cluster_batch,
                              all_encountered_seqs,
                              all_encountered_seq_occurrences,
                              all_encountered_context_occurrences,
                              all_encountered_rewards_punishments,
                              all_encountered_exploration_attempts,
                              all_encountered_exploration_results,
                              cluster_subbatch,
                              context_subbatch,
                              rewards_punishments_subbatch,
                              exploring_subbatch,
                              actions_subbatch)

    # Assert integrity of bookkeeping tensors
    assert same(expected_encountered_batch_occurrences, process.encountered_batch_seq_occurrences)
    assert same(expected_most_probable_batch_seq_probs, process.most_probable_batch_seq_probs)
    assert same(expected_most_probable_batch_seqs, process.most_probable_batch_seqs)
    assert same(expected_newly_encountered_seqs_counts, process.newly_encountered_seqs_counts)
    assert same(expected_newly_encountered_seqs_indicator, process.newly_encountered_seqs_indicator)
    assert same(expected_encountered_batch_exploration_attempts, process.encountered_batch_exploration_attempts)
    assert same(expected_encountered_batch_exploration_results,
                process.encountered_batch_exploration_results, eps=1e-7)

    # Assert the integrity of the return values
    assert same(expected_all_encountered_seqs, all_encountered_seqs)
    assert same(expected_all_encountered_occurrences, all_encountered_seq_occurrences)
    assert same(expected_all_encountered_context_occurrences, all_encountered_context_occurrences, eps=1e-4)
    assert same(expected_all_encountered_exploration_attempts, all_encountered_exploration_attempts)
    assert same(expected_all_encountered_exploration_results, all_encountered_exploration_results, eps=1e-2)
    assert same(expected_all_encountered_rewards_punishments, all_encountered_rewards_punishments, eps=1e-2)


def test_extract_info_known_seqs():
    device = 'cuda'
    float_dtype = get_float(device)
    seq_lookahead = 2
    seq_length = 3

    cluster_batch = torch.tensor([[[0.1, 0.9, 0],
                                   [0.8, 0, 0.2],
                                   [0, 1, 0],
                                   [1, 0, 0],
                                   [0.1, 0.8, 0.1]],
                                  [[0, 1, 0],
                                   [0, 0, 1],
                                   [1, 0, 0],
                                   [0.5, 0.5, 0],
                                   [0.1, 0.6, 0.3]]], dtype=float_dtype, device=device)

    rewards_punishments_batch = torch.tensor([[[0.1, 0.9],
                                               [0.8, 0],
                                               [0, 1],
                                               [1, 0],
                                               [0.1, 0.8]],
                                              [[0, 1],
                                               [0, 0],
                                               [1, 0],
                                               [0.5, 0.5],
                                               [0.1, 0.6]]], dtype=float_dtype, device=device)

    context_batch = torch.tensor([[[[1, 0, 1, 1], [0, 0, 0, 0]],
                                   [[0.2, 0.8, 0.33, 1], [0, 0, 0, 0]],
                                   [[0.6, 0.99, 1, 0], [0, 0, 0, 0]],
                                   [[0, 1, 1, 0], [0, 0, 0, 0]],
                                   [[0, 1, 1, 1], [0, 0, 0, 0]]],
                                  [[[1, 1, 1, 1], [0, 0, 0, 0]],
                                   [[0, 0, 0, 0], [0, 0, 0, 0]],
                                   [[1, 1, 0, 0], [0, 0, 0, 0]],
                                   [[0, 0, 1, 1], [0, 0, 0, 0]],
                                   [[0, 1, 0, 1], [0, 0, 0, 0]]]], dtype=float_dtype, device=device)

    exploring_batch = torch.tensor([[[1],
                                     [0],
                                     [1],
                                     [0],
                                     [1]],
                                    [[0],
                                     [1],
                                     [1],
                                     [1],
                                     [0]]], dtype=float_dtype, device=device)

    actions_batch = torch.tensor([[[0.5, 0.5, 0],
                                   [0, 0.2, 0.8],
                                   [1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]],
                                  [[0, 0.5, 0.5],
                                   [0, 0, 1],
                                   [0, 1, 0],
                                   [0, 0.5, 0.5],
                                   [0.2, 0.4, 0.4]]], dtype=float_dtype, device=device)

    all_encountered_seqs = torch.tensor([[[1, 0, 1],
                                          [2, 1, 0],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1]],
                                         [[1, 0, 1],
                                          [0, 1, 2],
                                          [1, 2, 0],
                                          [1, 2, 1],  # should be ignored because the occurrences == 0
                                          [2, 1, 2],  # should be ignored because it will be rewritten by the new seqs
                                          [-1, -1, -1],
                                          [-1, -1, -1]]], dtype=torch.int64, device=device)

    all_encountered_seq_occurrences = torch.tensor([[0.5, 0.5, 0, 0, 0, 0, 0],
                                                    [0.5, 0.5, 0.5, 0, 0.5, 0, 0]], dtype=float_dtype, device=device)

    process = create_tp_flock_learn_process(None,
                                            all_encountered_seqs,
                                            all_encountered_seq_occurrences,
                                            seq_length=seq_length,
                                            seq_lookahead=seq_lookahead,
                                            device=device)

    cluster_subbatch = cluster_batch.unsqueeze(1)
    context_subbatch = context_batch.unsqueeze(1)
    exploring_subbatch = exploring_batch.unsqueeze(1)
    actions_subbatch = actions_batch.unsqueeze(1)
    rewards_punishments_subbatch = rewards_punishments_batch.unsqueeze(1)

    process._extract_info_known_seqs(cluster_subbatch,
                                     context_subbatch,
                                     rewards_punishments_subbatch,
                                     exploring_subbatch,
                                     actions_subbatch,
                                     all_encountered_seqs,
                                     all_encountered_seq_occurrences,
                                     process.encountered_batch_seq_occurrences,
                                     process.encountered_batch_context_occurrences,
                                     process.encountered_batch_rewards_punishments,
                                     process.encountered_batch_exploration_attempts,
                                     process.encountered_batch_exploration_results,
                                     process.newly_encountered_seqs_indicator,
                                     process.encountered_subbatch_seq_occurrences,
                                     process.encountered_subbatch_context_occurrences,
                                     process.encountered_subbatch_rewards_punishments,
                                     process.encountered_subbatch_exploration_attempts,
                                     process.encountered_subbatch_exploration_results)

    check_cuda_errors()

    expected_encountered_batch_seq_occurrences = torch.tensor([[1.52, 0.2, 0, 0, 0, 0, 0],
                                                               [0, 0.15, 1, 0, 0, 0, 0]], dtype=float_dtype,
                                                              device=device)

    expected_newly_encountered_seqs_indicator = torch.tensor([[0, 1, 0],
                                                              [0, 1, 1]], dtype=torch.int64, device=device)

    assert same(expected_encountered_batch_seq_occurrences, process.encountered_batch_seq_occurrences)
    assert same(expected_newly_encountered_seqs_indicator, process.newly_encountered_seqs_indicator)

    # region Context
    expected_encountered_batch_context_occurrences = torch.tensor([[[[[1.2, 0.792, 1.52, 0.72]],
                                                                     [[0.144, 1.376, 1.0376, 0.72]],
                                                                     [[0.432, 1.5128, 1.52, 0.8]]],
                                                                    [[[0.04, 0.16, 0.066, 0.2]],
                                                                     [[0.12, 0.198, 0.2, 0]],
                                                                     [[0, 0.2, 0.2, 0]]],
                                                                    [[[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]]],
                                                                    [[[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]]],
                                                                    [[[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]]],
                                                                    [[[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]]],
                                                                    [[[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]]]],
                                                                   [[[[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]]],
                                                                    [[[0.15, 0.15, 0, 0]],
                                                                     [[0, 0, 0.15, 0.15]],
                                                                     [[0, 0.15, 0, 0.15]]],
                                                                    [[[1, 1, 1, 1]],
                                                                     [[0, 0, 0, 0]],
                                                                     [[1, 1, 0, 0]]],
                                                                    [[[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]]],
                                                                    [[[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]]],
                                                                    [[[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]]],
                                                                    [[[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]],
                                                                     [[0, 0, 0, 0]]]]],
                                                                  dtype=float_dtype, device=device)

    assert same(expected_encountered_batch_context_occurrences, process.encountered_batch_context_occurrences,
                eps=1e-3)

    # endregion

    # region exploration
    expected_encountered_batch_exploration_results = torch.tensor([[[[1.36, 0, 0.09],
                                                                     [0, 0, 0]],
                                                                    [[0, 0, 0],
                                                                     [0.2, 0, 0]],
                                                                    [[0, 0, 0],
                                                                     [0, 0, 0]],
                                                                    [[0, 0, 0],
                                                                     [0, 0, 0]],
                                                                    [[0, 0, 0],
                                                                     [0, 0, 0]],
                                                                    [[0, 0, 0],
                                                                     [0, 0, 0]],
                                                                    [[0, 0, 0],
                                                                     [0, 0, 0]]],
                                                                   [[[0, 0, 0],
                                                                     [0, 0, 0]],
                                                                    [[0.5, 0.5, 0.],
                                                                     [0.025, 0.15, 0.075]],
                                                                    [[0, 0, 0],
                                                                     [0, 0, 0]],
                                                                    [[0, 0, 0],
                                                                     [0, 0, 0]],
                                                                    [[0, 0, 0],
                                                                     [0, 0, 0]],
                                                                    [[0, 0, 0],
                                                                     [0, 0, 0]],
                                                                    [[0, 0, 0],
                                                                     [0, 0, 0]]]],
                                                                  dtype=float_dtype, device=device)

    expected_encountered_batch_exploration_attempts = torch.tensor([[[1.45, 0],
                                                                     [0, 0.2],
                                                                     [0, 0],
                                                                     [0, 0],
                                                                     [0, 0],
                                                                     [0, 0],
                                                                     [0, 0]],
                                                                    [[0, 0],
                                                                     [1., 0.25],
                                                                     [0, 0],
                                                                     [0, 0],
                                                                     [0, 0],
                                                                     [0, 0],
                                                                     [0, 0]]], dtype=float_dtype, device=device)

    assert same(expected_encountered_batch_exploration_results, process.encountered_batch_exploration_results, eps=1e-3)
    assert same(expected_encountered_batch_exploration_attempts, process.encountered_batch_exploration_attempts,
                eps=1e-3)

    # endregion

    # region rewards

    expected_encountered_batch_rewards_punishments = torch.tensor([[[[1.376, 0],
                                                                     [0.08, 1.36]],
                                                                    [[0, 0.2],
                                                                     [0.2, 0]],
                                                                    [[0, 0],
                                                                     [0, 0]],
                                                                    [[0, 0],
                                                                     [0, 0]],
                                                                    [[0, 0],
                                                                     [0, 0]],
                                                                    [[0, 0],
                                                                     [0, 0]],
                                                                    [[0, 0],
                                                                     [0, 0]]],
                                                                   [[[0, 0],
                                                                     [0, 0]],
                                                                    [[0.075, 0.075],
                                                                     [0.015, 0.09]],
                                                                    [[0, 0],
                                                                     [1, 0]],
                                                                    [[0, 0],
                                                                     [0, 0]],
                                                                    [[0, 0],
                                                                     [0, 0]],
                                                                    [[0, 0],
                                                                     [0, 0]],
                                                                    [[0, 0],
                                                                     [0, 0]]]], dtype=float_dtype, device=device)

    assert same(expected_encountered_batch_rewards_punishments, process.encountered_batch_rewards_punishments, eps=1e-3)
    # endregion


def test_update_knowledge_known_seqs():
    device = 'cpu'
    float_dtype = get_float(device)
    all_encountered_seq_occurrences = torch.tensor([[0.5, 0.5, 0, 0, 0, 0, 0],
                                                    [0.5, 0.5, 0.5, 0, 0, 0, 0]], dtype=float_dtype)
    encountered_batch_seq_occurrences = torch.tensor([[1.52, 0.2, 0, 0, 0, 0, 0],
                                                      [0, 0.15, 1, 0, 0, 0, 0]], dtype=float_dtype)

    all_encountered_context_occurrences = torch.tensor([[[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                                                         [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                                                         [[1, 2, 3], [1, 2, 3], [1, 2, 3]]],
                                                        [[[4, 4, 4], [4, 4, 4], [4, 4, 4]],
                                                         [[4, 4, 4], [4, 4, 4], [4, 4, 4]],
                                                         [[4, 4, 4], [4, 4, 4], [4, 4, 4]]]], dtype=float_dtype)

    encountered_batch_context_occurrences = torch.tensor([[[[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
                                                           [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
                                                           [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]],
                                                          [[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
                                                           [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
                                                           [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]]],
                                                         dtype=float_dtype)

    all_encountered_exploration_attempts = torch.tensor([[[3],
                                                          [5.2],
                                                          [0],
                                                          [0],
                                                          [0],
                                                          [0],
                                                          [0]],
                                                         [[0],
                                                          [0],
                                                          [1],
                                                          [0],
                                                          [0],
                                                          [0],
                                                          [0]]], dtype=float_dtype)

    encountered_batch_exploration_attempts = torch.tensor([[[1.5],
                                                            [0],
                                                            [0],
                                                            [0],
                                                            [0],
                                                            [0],
                                                            [0]],
                                                           [[1],
                                                            [2],
                                                            [3],
                                                            [0],
                                                            [0],
                                                            [0],
                                                            [0]]], dtype=float_dtype)

    all_encountered_exploration_results = torch.tensor([[[[0.1, 0, 0]],
                                                         [[0.5, 0, 0]],
                                                         [[0, 0, 0]],
                                                         [[0, 0, 0]],
                                                         [[0, 0, 0]],
                                                         [[0, 0, 0]],
                                                         [[0, 0, 0]]],
                                                        [[[0, 0, 0]],
                                                         [[0, 0, 0]],
                                                         [[1, 0, 0]],
                                                         [[0, 0, 0]],
                                                         [[0, 0, 0]],
                                                         [[0, 0, 0]],
                                                         [[0, 0, 0]]]], dtype=float_dtype)

    encountered_batch_exploration_results = torch.tensor([[[[1.4, 0, 0]],
                                                           [[0, 0, 0]],
                                                           [[0, 0, 0]],
                                                           [[0, 0, 0]],
                                                           [[0, 0, 0]],
                                                           [[0, 0, 0]],
                                                           [[0, 0, 0]]],
                                                          [[[1, 0, 0]],
                                                           [[0, 0, 0]],
                                                           [[0.5, 0, 0]],
                                                           [[0, 0, 0]],
                                                           [[0, 0, 0]],
                                                           [[0, 0, 0]],
                                                           [[0, 0, 0]]]], dtype=float_dtype)

    all_encountered_rewards_punishments = torch.tensor([[[[0.1, 0]],
                                                         [[0.5, 0]],
                                                         [[0, 0]],
                                                         [[0, 0]],
                                                         [[0, 0]],
                                                         [[0, 0]],
                                                         [[0, 0]]],
                                                        [[[0, 0]],
                                                         [[0, 0]],
                                                         [[1, 0]],
                                                         [[0, 0]],
                                                         [[0, 0]],
                                                         [[0, 0]],
                                                         [[0, 0]]]], dtype=float_dtype)

    encountered_batch_rewards_punishments = torch.tensor([[[[1.4, 0]],
                                                           [[0, 0]],
                                                           [[0, 0]],
                                                           [[0, 0]],
                                                           [[0, 0]],
                                                           [[0, 0]],
                                                           [[0, 0]]],
                                                          [[[1, 0]],
                                                           [[0, 0]],
                                                           [[0.5, 0]],
                                                           [[0, 0]],
                                                           [[0, 0]],
                                                           [[0, 0]],
                                                           [[0, 0]]]], dtype=float_dtype)

    TPFlockLearning._update_knowledge_known_seqs(all_encountered_seq_occurrences,
                                                 all_encountered_context_occurrences,
                                                 all_encountered_rewards_punishments,
                                                 all_encountered_exploration_attempts,
                                                 all_encountered_exploration_results,
                                                 encountered_batch_seq_occurrences,
                                                 encountered_batch_context_occurrences,
                                                 encountered_batch_rewards_punishments,
                                                 encountered_batch_exploration_attempts,
                                                 encountered_batch_exploration_results)

    expected_all_encountered_seq_occurrences = torch.tensor([[2.02, 0.7, 0, 0, 0, 0, 0],
                                                             [0.5, 0.65, 1.5, 0, 0, 0, 0]], dtype=float_dtype)

    expected_all_encountered_context_occurrences = torch.tensor([[[[1.2, 2.2, 3.2], [1.2, 2.2, 3.2], [1.2, 2.2, 3.2]],
                                                                  [[1.2, 2.2, 3.2], [1.2, 2.2, 3.2], [1.2, 2.2, 3.2]],
                                                                  [[1.2, 2.2, 3.2], [1.2, 2.2, 3.2], [1.2, 2.2, 3.2]]],
                                                                 [[[4.1, 4.2, 4.3], [4.1, 4.2, 4.3], [4.1, 4.2, 4.3]],
                                                                  [[4.1, 4.2, 4.3], [4.1, 4.2, 4.3], [4.1, 4.2, 4.3]],
                                                                  [[4.1, 4.2, 4.3], [4.1, 4.2, 4.3], [4.1, 4.2, 4.3]]]],
                                                                dtype=float_dtype)

    expected_all_encountered_exploration_attempts = torch.tensor([[[4.5],
                                                                   [5.2],
                                                                   [0],
                                                                   [0],
                                                                   [0],
                                                                   [0],
                                                                   [0]],
                                                                  [[1],
                                                                   [2],
                                                                   [4],
                                                                   [0],
                                                                   [0],
                                                                   [0],
                                                                   [0]]], dtype=float_dtype)

    expected_all_encountered_exploration_results = torch.tensor([[[[0.3778, 0, 0]],
                                                                  [[0.5, 0, 0]],
                                                                  [[0, 0, 0]],
                                                                  [[0, 0, 0]],
                                                                  [[0, 0, 0]],
                                                                  [[0, 0, 0]],
                                                                  [[0, 0, 0]]],
                                                                 [[[1, 0, 0]],
                                                                  [[0, 0, 0]],
                                                                  [[0.375, 0, 0]],
                                                                  [[0, 0, 0]],
                                                                  [[0, 0, 0]],
                                                                  [[0, 0, 0]],
                                                                  [[0, 0, 0]]]], dtype=float_dtype)

    expected_all_encountered_rewards_punishments = torch.tensor([[[[1.5, 0]],
                                                                  [[0.5, 0]],
                                                                  [[0, 0]],
                                                                  [[0, 0]],
                                                                  [[0, 0]],
                                                                  [[0, 0]],
                                                                  [[0, 0]]],
                                                                 [[[1, 0]],
                                                                  [[0, 0]],
                                                                  [[1.5, 0]],
                                                                  [[0, 0]],
                                                                  [[0, 0]],
                                                                  [[0, 0]],
                                                                  [[0, 0]]]], dtype=float_dtype)

    assert same(expected_all_encountered_seq_occurrences, all_encountered_seq_occurrences)
    assert same(expected_all_encountered_context_occurrences, all_encountered_context_occurrences)
    assert same(expected_all_encountered_exploration_attempts, all_encountered_exploration_attempts)
    assert same(expected_all_encountered_exploration_results, all_encountered_exploration_results, eps=1e-4)
    assert same(expected_all_encountered_rewards_punishments, all_encountered_rewards_punishments)


def test_identify_new_seqs():
    device = 'cuda'
    float_dtype = get_float(device)

    batch = torch.tensor([[[0.1, 0.9, 0],
                           [0.8, 0, 0.2],
                           [0, 1, 0],
                           [1, 0, 0],
                           [0.1, 0.8, 0.1]],
                          [[0, 1, 0],
                           [0, 0, 1],
                           [1, 0, 0],
                           [0.5, 0.5, 0],
                           [0.1, 0.6, 0.3]]], dtype=float_dtype, device=device)

    newly_encountered_seqs_indicator = torch.tensor([[0, 1, 0],
                                                     [1, 1, 0]], dtype=torch.int64, device=device)

    process = create_tp_flock_learn_process(device=device)

    process.most_probable_batch_seqs.fill_(-1)
    process.most_probable_batch_seq_probs.fill_(-1)

    process._identify_new_seqs(batch,
                               newly_encountered_seqs_indicator,
                               process.most_probable_batch_seqs,
                               process.most_probable_batch_seq_probs)

    check_cuda_errors()

    expected_most_probable_batch_seq_probs = torch.tensor([[0, 0.8, 0],
                                                           [1, 0, 0]], dtype=float_dtype, device=device)
    expected_most_probable_batch_seqs = torch.tensor([[[1, 0, 1],
                                                       [0, 1, 0],
                                                       [1, 0, 1]],
                                                      [[1, 2, 0],
                                                       [2, 0, 0],
                                                       [0, 0, 1]]], dtype=torch.int64, device=device)

    assert same(process.most_probable_batch_seq_probs, expected_most_probable_batch_seq_probs)
    assert same(process.most_probable_batch_seqs, expected_most_probable_batch_seqs)


@pytest.mark.parametrize(
    "most_probable_batch_seq_probs, most_probable_batch_seqs, expected_newly_encountered_seqs_counts, expected_most_probable_batch_seqs",
    [([[0.6, 0, 0.8], [1, 0.6, 0]], [[[1, 0, 1], [0, 1, 0], [1, 0, 1]], [[1, 2, 0], [2, 0, 1], [0, 0, 1]]],
      [[0, 0, 1.4], [0, 1, 0.6]], [[[-1, -1, -1], [-1, -1, -1], [1, 0, 1]], [[-1, -1, -1], [1, 2, 0], [2, 0, 1]]]),

     # All invalid seq
     ([[0, 0, 0], [0, 0, 0]], [[[1, 1, 1], [2, 2, 0], [0, 0, 1]], [[1, 0, 0], [2, 1, 1], [0, 0, 1]]],
      [[0, 0, 0], [0, 0, 0]], [[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]], [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]),

     # All same seq
     ([[1, 1, 1], [1, 1, 1]], [[[0, 1, 0], [0, 1, 0], [0, 1, 0]], [[2, 1, 0], [2, 1, 0], [2, 1, 0]]],
      [[0, 0, 3], [0, 0, 3]], [[[-1, -1, -1], [-1, -1, -1], [0, 1, 0]], [[-1, -1, -1], [-1, -1, -1], [2, 1, 0]]])])
def test_extract_info_new_seqs(most_probable_batch_seq_probs, most_probable_batch_seqs,
                               expected_newly_encountered_seqs_counts, expected_most_probable_batch_seqs):
    device = 'cuda'
    float_dtype = get_float(device)

    process = create_tp_flock_learn_process(device=device)

    process.most_probable_batch_seq_probs = torch.tensor(most_probable_batch_seq_probs, dtype=float_dtype,
                                                         device=device)
    process.most_probable_batch_seqs = torch.tensor(most_probable_batch_seqs, dtype=torch.int64, device=device)

    process.newly_encountered_seqs_counts.fill_(-1)

    process._extract_info_new_seqs(process.most_probable_batch_seqs,
                                   process.most_probable_batch_seq_probs,
                                   process.newly_encountered_seqs_counts)

    check_cuda_errors()

    expected_newly_encountered_seqs_counts = torch.tensor(expected_newly_encountered_seqs_counts, dtype=float_dtype,
                                                          device=device)

    expected_most_probable_batch_seqs = torch.tensor(expected_most_probable_batch_seqs, dtype=torch.int64,
                                                     device=device)

    assert same(expected_newly_encountered_seqs_counts, process.newly_encountered_seqs_counts, eps=1e-6)
    assert same(expected_most_probable_batch_seqs, process.most_probable_batch_seqs)


# non-default stream is not currently tested, so here is just the default one
@pytest.mark.parametrize('stream', [torch.cuda.current_stream()])
def test_erase_unseen_seqs(stream):
    with torch.cuda.stream(stream):
        proc = create_tp_flock_learn_process()
        device = 'cuda'
        float_dtype = get_float(device)

        most_probable_batch_seqs = torch.tensor([[[1, 0, 1],
                                                  [0, 1, 0],
                                                  [1, 0, 1]],
                                                 [[1, 2, 0],
                                                  [2, 0, 1],
                                                  [0, 0, 1]]], dtype=torch.int64, device=device)

        newly_encountered_seqs_counts = torch.tensor([[1.4, 0, 0],
                                                      [1, 0.6, 0]], dtype=float_dtype, device=device)

        proc._erase_unseen_seqs(most_probable_batch_seqs, newly_encountered_seqs_counts)

        expected_most_probable_batch_seqs = torch.tensor([[[1, 0, 1],
                                                           [-1, -1, -1],
                                                           [-1, -1, -1]],
                                                          [[1, 2, 0],
                                                           [2, 0, 1],
                                                           [-1, -1, -1]]], dtype=torch.int64, device=device)

        assert same(expected_most_probable_batch_seqs, most_probable_batch_seqs)

        # Test single flock
        proc = create_tp_flock_learn_process()

        most_probable_batch_seqs = torch.tensor([[[1, 0, 1, 3],
                                                  [0, 1, 0, 2],
                                                  [1, 0, 1, 7]]], dtype=torch.int64, device=device)

        newly_encountered_seqs_counts = torch.tensor([[1.4, 0, 0]], dtype=float_dtype, device=device)

        proc._erase_unseen_seqs(most_probable_batch_seqs, newly_encountered_seqs_counts)

        expected_most_probable_batch_seqs = torch.tensor([[[1, 0, 1, 3],
                                                           [-1, -1, -1, -1],
                                                           [-1, -1, -1, -1]]], dtype=torch.int64, device=device)

        assert same(expected_most_probable_batch_seqs, most_probable_batch_seqs)


@pytest.mark.skip(reason="This test is here because it failed on the non-default stream. "
                         "Currently, it is not run because it is slow.")
@pytest.mark.parametrize('stream', [torch.cuda.current_stream()])
def test_erase_unseen_seqs_real_data(stream):
    with torch.cuda.stream(stream):
        device = 'cuda'
        float_dtype = get_float(device)

        # Test real data - those data failed on the non-default stream
        proc = create_tp_flock_learn_process()

        # region real data
        most_probable_batch_seqs = torch.tensor([[[2, 7, 17, 10],
                                                  [7, 17, 10, 2],
                                                  [17, 10, 2, 11],
                                                  [10, 2, 11, 2],
                                                  [2, 11, 2, 17],
                                                  [11, 2, 17, 10],
                                                  [2, 17, 10, 2],
                                                  [17, 10, 2, 13],
                                                  [10, 2, 13, 17],
                                                  [2, 13, 17, 2],
                                                  [13, 17, 2, 17],
                                                  [17, 2, 17, 13],
                                                  [2, 17, 13, 10],
                                                  [17, 13, 10, 7],
                                                  [13, 10, 7, 11],
                                                  [10, 7, 11, 1],
                                                  [7, 11, 1, 10],
                                                  [11, 1, 10, 7],
                                                  [1, 10, 7, 10],
                                                  [10, 7, 10, 7],
                                                  [7, 10, 7, 11],
                                                  [10, 7, 11, 13],
                                                  [7, 11, 13, 2],
                                                  [11, 13, 2, 10],
                                                  [13, 2, 10, 5],
                                                  [2, 10, 5, 10],
                                                  [10, 5, 10, 11],
                                                  [5, 10, 11, 7],
                                                  [10, 11, 7, 11],
                                                  [11, 7, 11, 14],
                                                  [7, 11, 14, 11],
                                                  [11, 14, 11, 7],
                                                  [14, 11, 7, 10],
                                                  [11, 7, 10, 2],
                                                  [7, 10, 2, 7],
                                                  [10, 2, 7, 14],
                                                  [2, 7, 14, 13],
                                                  [7, 14, 13, 10],
                                                  [14, 13, 10, 7],
                                                  [13, 10, 7, 11],
                                                  [10, 7, 11, 7],
                                                  [7, 11, 7, 10],
                                                  [11, 7, 10, 13],
                                                  [7, 10, 13, 7],
                                                  [10, 13, 7, 12],
                                                  [13, 7, 12, 2],
                                                  [7, 12, 2, 13],
                                                  [12, 2, 13, 17],
                                                  [2, 13, 17, 2],
                                                  [13, 17, 2, 10],
                                                  [17, 2, 10, 7],
                                                  [2, 10, 7, 11],
                                                  [10, 7, 11, 12],
                                                  [7, 11, 12, 13],
                                                  [11, 12, 13, 14],
                                                  [12, 13, 14, 7],
                                                  [13, 14, 7, 14],
                                                  [14, 7, 14, 1],
                                                  [7, 14, 1, 2],
                                                  [14, 1, 2, 10],
                                                  [1, 2, 10, 13],
                                                  [2, 10, 13, 11],
                                                  [10, 13, 11, 17],
                                                  [13, 11, 17, 13],
                                                  [11, 17, 13, 10],
                                                  [17, 13, 10, 7],
                                                  [13, 10, 7, 11],
                                                  [10, 7, 11, 5],
                                                  [7, 11, 5, 13],
                                                  [11, 5, 13, 14],
                                                  [5, 13, 14, 13],
                                                  [13, 14, 13, 2],
                                                  [14, 13, 2, 11],
                                                  [13, 2, 11, 2],
                                                  [2, 11, 2, 12],
                                                  [11, 2, 12, 2],
                                                  [2, 12, 2, 17],
                                                  [12, 2, 17, 8],
                                                  [2, 17, 8, 2],
                                                  [17, 8, 2, 10],
                                                  [8, 2, 10, 13],
                                                  [2, 10, 13, 17],
                                                  [10, 13, 17, 11],
                                                  [13, 17, 11, 13],
                                                  [17, 11, 13, 12],
                                                  [11, 13, 12, 5],
                                                  [13, 12, 5, 2],
                                                  [12, 5, 2, 10],
                                                  [5, 2, 10, 13],
                                                  [2, 10, 13, 7],
                                                  [10, 13, 7, 13],
                                                  [13, 7, 13, 11],
                                                  [7, 13, 11, 12],
                                                  [13, 11, 12, 1],
                                                  [11, 12, 1, 7],
                                                  [12, 1, 7, 11],
                                                  [1, 7, 11, 13],
                                                  [7, 11, 13, 7],
                                                  [11, 13, 7, 5],
                                                  [13, 7, 5, 7],
                                                  [7, 5, 7, 13],
                                                  [5, 7, 13, 7],
                                                  [7, 13, 7, 17],
                                                  [13, 7, 17, 14],
                                                  [7, 17, 14, 17],
                                                  [17, 14, 17, 7],
                                                  [14, 17, 7, 13],
                                                  [17, 7, 13, 11],
                                                  [7, 13, 11, 10],
                                                  [13, 11, 10, 13],
                                                  [11, 10, 13, 7],
                                                  [10, 13, 7, 12],
                                                  [13, 7, 12, 17],
                                                  [7, 12, 17, 7],
                                                  [12, 17, 7, 11],
                                                  [17, 7, 11, 13],
                                                  [7, 11, 13, 7],
                                                  [11, 13, 7, 11],
                                                  [13, 7, 11, 13],
                                                  [7, 11, 13, 7],
                                                  [11, 13, 7, 11],
                                                  [13, 7, 11, 10],
                                                  [7, 11, 10, 8],
                                                  [11, 10, 8, 4],
                                                  [10, 8, 4, 14],
                                                  [8, 4, 14, 11],
                                                  [4, 14, 11, 1],
                                                  [14, 11, 1, 14],
                                                  [11, 1, 14, 13],
                                                  [1, 14, 13, 7],
                                                  [14, 13, 7, 13],
                                                  [13, 7, 13, 2],
                                                  [7, 13, 2, 7],
                                                  [13, 2, 7, 5],
                                                  [2, 7, 5, 7],
                                                  [7, 5, 7, 2],
                                                  [5, 7, 2, 7],
                                                  [7, 2, 7, 13],
                                                  [2, 7, 13, 7],
                                                  [7, 13, 7, 11],
                                                  [13, 7, 11, 2],
                                                  [7, 11, 2, 10],
                                                  [11, 2, 10, 11],
                                                  [2, 10, 11, 1],
                                                  [10, 11, 1, 10],
                                                  [11, 1, 10, 7],
                                                  [1, 10, 7, 13],
                                                  [10, 7, 13, 5],
                                                  [7, 13, 5, 2],
                                                  [13, 5, 2, 13],
                                                  [5, 2, 13, 7],
                                                  [2, 13, 7, 10],
                                                  [13, 7, 10, 2],
                                                  [7, 10, 2, 7],
                                                  [10, 2, 7, 13],
                                                  [2, 7, 13, 2],
                                                  [7, 13, 2, 7],
                                                  [13, 2, 7, 2],
                                                  [2, 7, 2, 13],
                                                  [7, 2, 13, 2],
                                                  [2, 13, 2, 13],
                                                  [13, 2, 13, 10],
                                                  [2, 13, 10, 11],
                                                  [13, 10, 11, 13],
                                                  [10, 11, 13, 17],
                                                  [11, 13, 17, 2],
                                                  [13, 17, 2, 13],
                                                  [17, 2, 13, 2],
                                                  [2, 13, 2, 13],
                                                  [13, 2, 13, 7],
                                                  [2, 13, 7, 2],
                                                  [13, 7, 2, 10],
                                                  [7, 2, 10, 2],
                                                  [2, 10, 2, 17],
                                                  [10, 2, 17, 2],
                                                  [2, 17, 2, 13],
                                                  [17, 2, 13, 7],
                                                  [2, 13, 7, 2],
                                                  [13, 7, 2, 5],
                                                  [7, 2, 5, 11],
                                                  [2, 5, 11, 7],
                                                  [5, 11, 7, 13],
                                                  [11, 7, 13, 2],
                                                  [7, 13, 2, 11],
                                                  [13, 2, 11, 7],
                                                  [2, 11, 7, 8],
                                                  [11, 7, 8, 7],
                                                  [7, 8, 7, 17],
                                                  [8, 7, 17, 5],
                                                  [7, 17, 5, 14],
                                                  [17, 5, 14, 11],
                                                  [5, 14, 11, 5],
                                                  [14, 11, 5, 17],
                                                  [11, 5, 17, 7],
                                                  [5, 17, 7, 11],
                                                  [17, 7, 11, 7],
                                                  [7, 11, 7, 13],
                                                  [11, 7, 13, 7],
                                                  [7, 13, 7, 13],
                                                  [13, 7, 13, 11],
                                                  [7, 13, 11, 13],
                                                  [13, 11, 13, 10],
                                                  [11, 13, 10, 17],
                                                  [13, 10, 17, 10],
                                                  [10, 17, 10, 7],
                                                  [17, 10, 7, 1],
                                                  [10, 7, 1, 11],
                                                  [7, 1, 11, 7],
                                                  [1, 11, 7, 8],
                                                  [11, 7, 8, 1],
                                                  [7, 8, 1, 13],
                                                  [8, 1, 13, 7],
                                                  [1, 13, 7, 10],
                                                  [13, 7, 10, 7],
                                                  [7, 10, 7, 5],
                                                  [10, 7, 5, 7],
                                                  [7, 5, 7, 13],
                                                  [5, 7, 13, 7],
                                                  [7, 13, 7, 13],
                                                  [13, 7, 13, 12],
                                                  [7, 13, 12, 2],
                                                  [13, 12, 2, 10],
                                                  [12, 2, 10, 2],
                                                  [2, 10, 2, 13],
                                                  [10, 2, 13, 7],
                                                  [2, 13, 7, 5],
                                                  [13, 7, 5, 13],
                                                  [7, 5, 13, 7],
                                                  [5, 13, 7, 13],
                                                  [13, 7, 13, 7],
                                                  [7, 13, 7, 14],
                                                  [13, 7, 14, 13],
                                                  [7, 14, 13, 1],
                                                  [14, 13, 1, 14],
                                                  [13, 1, 14, 11],
                                                  [1, 14, 11, 2],
                                                  [14, 11, 2, 13],
                                                  [11, 2, 13, 12],
                                                  [2, 13, 12, 10],
                                                  [13, 12, 10, 13],
                                                  [12, 10, 13, 11],
                                                  [10, 13, 11, 10],
                                                  [13, 11, 10, 17],
                                                  [11, 10, 17, 10],
                                                  [10, 17, 10, 13],
                                                  [17, 10, 13, 7],
                                                  [10, 13, 7, 13],
                                                  [13, 7, 13, 17],
                                                  [7, 13, 17, 10],
                                                  [13, 17, 10, 13],
                                                  [17, 10, 13, 11],
                                                  [10, 13, 11, 13],
                                                  [13, 11, 13, 7],
                                                  [11, 13, 7, 10],
                                                  [13, 7, 10, 14],
                                                  [7, 10, 14, 10],
                                                  [10, 14, 10, 12],
                                                  [14, 10, 12, 2],
                                                  [10, 12, 2, 17],
                                                  [12, 2, 17, 2],
                                                  [2, 17, 2, 10],
                                                  [17, 2, 10, 7],
                                                  [2, 10, 7, 2],
                                                  [10, 7, 2, 13],
                                                  [7, 2, 13, 12],
                                                  [2, 13, 12, 7],
                                                  [13, 12, 7, 12],
                                                  [12, 7, 12, 2],
                                                  [7, 12, 2, 13],
                                                  [12, 2, 13, 2],
                                                  [2, 13, 2, 13],
                                                  [13, 2, 13, 10],
                                                  [2, 13, 10, 17],
                                                  [13, 10, 17, 7],
                                                  [10, 17, 7, 13],
                                                  [17, 7, 13, 12],
                                                  [7, 13, 12, 7],
                                                  [13, 12, 7, 10],
                                                  [12, 7, 10, 12],
                                                  [7, 10, 12, 10],
                                                  [10, 12, 10, 7],
                                                  [12, 10, 7, 13],
                                                  [10, 7, 13, 7],
                                                  [7, 13, 7, 11],
                                                  [13, 7, 11, 2],
                                                  [7, 11, 2, 12],
                                                  [11, 2, 12, 17],
                                                  [2, 12, 17, 11],
                                                  [12, 17, 11, 17],
                                                  [17, 11, 17, 2],
                                                  [11, 17, 2, 10],
                                                  [17, 2, 10, 7],
                                                  [2, 10, 7, 17],
                                                  [10, 7, 17, 5],
                                                  [7, 17, 5, 7],
                                                  [17, 5, 7, 13],
                                                  [5, 7, 13, 11]]], device=device, dtype=torch.int64)

        newly_encountered_seqs_counts = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1., 1.,
                                                       1., 2., 3., 1., 1., 2., 1., 1., 1., 1., 1., 1.,
                                                       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1.,
                                                       1., 1., 1., 0., 1., 1., 1., 1., 2., 1., 2., 1.,
                                                       0., 1., 3., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                       1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1.,
                                                       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                       1., 1., 1., 1., 1., 1., 2., 2., 1., 1., 1., 1.,
                                                       1., 3., 1., 1., 2., 2., 1., 1., 1., 1., 1., 1.,
                                                       1., 1., 1., 0., 1., 1., 1., 1., 0., 2., 1., 0.,
                                                       0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                       2., 1., 1., 1., 1., 1., 1., 2., 2., 1., 1., 1.,
                                                       1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1.,
                                                       0., 1., 1., 1., 3., 2., 1., 1., 1., 1., 1., 1.,
                                                       0., 1., 2., 1., 1., 1., 1., 1., 1., 0., 1., 1.,
                                                       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                       1., 1., 1., 1., 1., 1., 2., 0., 1., 1., 1., 1.,
                                                       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                       0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                       1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.,
                                                       1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1.,
                                                       1., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1.,
                                                       1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1.,
                                                       1., 1., 1., 0., 1., 1., 1., 1., 1.]], device=device)
        # endregion
        expected_most_probable_batch_seqs = most_probable_batch_seqs.clone()

        expected_most_probable_batch_seqs.view(-1, 4)[
            newly_encountered_seqs_counts.view(-1) == 0] = -1

        proc._erase_unseen_seqs(most_probable_batch_seqs, newly_encountered_seqs_counts)
        assert same(expected_most_probable_batch_seqs, most_probable_batch_seqs)


def test_update_knowledge_new_seqs():
    device = 'cpu'
    float_dtype = get_float(device)

    all_encountered_seqs = torch.tensor([[[1, 0, 1],
                                          [2, 1, 0],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1]],
                                         [[1, 0, 1],
                                          [0, 1, 2],
                                          [1, 2, 0],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1]]], dtype=torch.int64, device=device)

    all_encountered_seq_occurrences = torch.tensor([[0.5, 0.5, 0, 0, 0, 0, 0],
                                                    [0.5, 0.5, 0.5, 0, 0, 0, 0]], dtype=float_dtype, device=device)

    all_encountered_context_occurrences = torch.tensor([[[[[2, 2]], [[2, 2]], [[2, 2]]],
                                                         [[[1, 1]], [[1, 1]], [[1, 1]]],
                                                         [[[0., 0.]], [[0., 2.]], [[0., 0.]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[3., 3.]], [[3., 3.]], [[3., 3.]]],
                                                         [[[1., 1]], [[1., 1.]], [[1., 1.]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]]],
                                                        [[[[3, 3]], [[3, 3]], [[3, 3]]],
                                                         [[[4, 4]], [[4, 4]], [[4, 4]]],
                                                         [[[5, 5]], [[5, 5]], [[5, 5]]],
                                                         [[[0., 0.]], [[1., 0.]], [[0., 0.]]],
                                                         [[[3.3, 3.3]], [[3., 6.]], [[3., 3.]]],
                                                         [[[3.4, 3.1]], [[3., 3.]], [[9., 0.]]],
                                                         [[[3.1, 3.]], [[7., 2.]], [[3.23, 1.]]]]], dtype=float_dtype,
                                                       device=device)

    all_encountered_exploration_attempts = torch.tensor([[[3],
                                                          [5.2],
                                                          [0],
                                                          [1],
                                                          [2],
                                                          [0],
                                                          [0]],
                                                         [[0],
                                                          [0],
                                                          [1],
                                                          [1],
                                                          [2],
                                                          [3],
                                                          [4]]], dtype=float_dtype)

    all_encountered_exploration_results = torch.tensor([[[[3, 3, 3]],
                                                         [[5.2, 5.2, 5.2]],
                                                         [[0, 0, 0]],
                                                         [[1, 1, 1]],
                                                         [[2, 2, 2]],
                                                         [[0, 0, 0]],
                                                         [[0, 0, 0]]],
                                                        [[[0, 0, 0]],
                                                         [[0, 0, 0]],
                                                         [[1, 1, 1]],
                                                         [[2, 2, 2]],
                                                         [[3, 3, 3]],
                                                         [[4, 4, 4]],
                                                         [[5, 5, 5]]]], dtype=float_dtype)

    all_encountered_rewards_punishments = torch.tensor([[[[3, 3]],
                                                         [[5.2, 5.2]],
                                                         [[0, 0]],
                                                         [[1, 1]],
                                                         [[2, 2]],
                                                         [[0, 0]],
                                                         [[0, 0]]],
                                                        [[[0, 0]],
                                                         [[0, 0]],
                                                         [[1, 1]],
                                                         [[2, 2]],
                                                         [[3, 3]],
                                                         [[4, 4]],
                                                         [[5, 5]]]], dtype=float_dtype)

    process = create_tp_flock_learn_process(flock_size=2, max_encountered_seqs=7, seq_length=3,
                                            context_size=2, device=device, n_cluster_centers=3,
                                            all_encountered_context_occurrences=all_encountered_context_occurrences,
                                            all_encountered_seq_occurrences=all_encountered_seq_occurrences,
                                            all_encountered_seqs=all_encountered_seqs)

    process.most_probable_batch_seqs = torch.tensor([[[0, 1, 0],
                                                      [-1, -1, -1],
                                                      [-1, -1, -1]],
                                                     [[1, 2, 1],
                                                      [2, 0, 1],
                                                      [-1, -1, -1]]], dtype=torch.int64, device=device)

    process.newly_encountered_seqs_counts = torch.tensor([[1.4, 0, 0],
                                                          [1, 0.6, 0]], dtype=float_dtype, device=device)

    process._update_knowledge_new_seqs(all_encountered_seqs,
                                       all_encountered_seq_occurrences,
                                       all_encountered_context_occurrences,
                                       all_encountered_rewards_punishments,
                                       all_encountered_exploration_attempts,
                                       all_encountered_exploration_results,
                                       process.most_probable_batch_seqs,
                                       process.newly_encountered_seqs_counts)

    expected_all_encountered_seqs = torch.tensor([[[1, 0, 1],
                                                   [2, 1, 0],
                                                   [-1, -1, -1],
                                                   [-1, -1, -1],
                                                   [0, 1, 0],
                                                   [-1, -1, -1],
                                                   [-1, -1, -1]],
                                                  [[1, 0, 1],
                                                   [0, 1, 2],
                                                   [1, 2, 0],
                                                   [-1, -1, -1],
                                                   [1, 2, 1],
                                                   [2, 0, 1],
                                                   [-1, -1, -1]]], dtype=torch.int64)

    expected_all_encountered_seq_occurrences = torch.tensor([[0.5, 0.5, 0, 0, 1.4, 0, 0],
                                                             [0.5, 0.5, 0.5, 0, 1, 0.6, 0]], dtype=float_dtype)

    expected_all_encountered_context_occurrences = torch.tensor([[[[[2, 2]], [[2, 2]], [[2, 2]]],
                                                                  [[[1, 1]], [[1, 1]], [[1, 1]]],
                                                                  [[[0., 0.]], [[0., 2.]], [[0., 0.]]],
                                                                  [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                                  [[[0.7, 0.7]], [[0.7, 0.7]], [[0.7, 0.7]]],
                                                                  [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                                  [[[0., 0.]], [[0., 0.]], [[0., 0.]]]],
                                                                 [[[[3, 3]], [[3, 3]], [[3, 3]]],
                                                                  [[[4, 4]], [[4, 4]], [[4, 4]]],
                                                                  [[[5, 5]], [[5, 5]], [[5, 5]]],
                                                                  [[[0., 0.]], [[1., 0.]], [[0., 0.]]],
                                                                  [[[0.5, 0.5]], [[0.5, 0.5]], [[0.5, 0.5]]],
                                                                  [[[0.3, 0.3]], [[0.3, 0.3]], [[0.3, 0.3]]],
                                                                  [[[0., 0.]], [[0., 0.]], [[0., 0.]]]]],
                                                                dtype=float_dtype,
                                                                device=device)

    prior = DEFAULT_EXPLORATION_ATTEMPTS_PRIOR
    expected_all_encountered_exploration_attempts = torch.tensor([[[3],
                                                                   [5.2],
                                                                   [0],
                                                                   [1],
                                                                   [prior],
                                                                   [prior],
                                                                   [prior]],
                                                                  [[0],
                                                                   [0],
                                                                   [1],
                                                                   [1],
                                                                   [prior],
                                                                   [prior],
                                                                   [prior]]], dtype=float_dtype)

    prior = 0
    expected_all_encountered_exploration_results = torch.tensor([[[[3, 3, 3]],
                                                                  [[5.2, 5.2, 5.2]],
                                                                  [[0, 0, 0]],
                                                                  [[1, 1, 1]],
                                                                  [[1, prior, prior]],
                                                                  [[prior, prior, prior]],
                                                                  [[prior, prior, prior]]],
                                                                 [[[0, 0, 0]],
                                                                  [[0, 0, 0]],
                                                                  [[1, 1, 1]],
                                                                  [[2, 2, 2]],
                                                                  [[prior, 1, prior]],
                                                                  [[prior, 1, prior]],
                                                                  [[prior, prior, prior]]]], dtype=float_dtype)

    expected_all_encountered_rewards_punishments = torch.tensor([[[[3, 3]],
                                                                  [[5.2, 5.2]],
                                                                  [[0, 0]],
                                                                  [[1, 1]],
                                                                  [[0, 0]],
                                                                  [[0, 0]],
                                                                  [[0, 0]]],
                                                                 [[[0, 0]],
                                                                  [[0, 0]],
                                                                  [[1, 1]],
                                                                  [[2, 2]],
                                                                  [[0, 0]],
                                                                  [[0, 0]],
                                                                  [[0, 0]]]], dtype=float_dtype)

    assert same(expected_all_encountered_seqs, all_encountered_seqs)
    assert same(expected_all_encountered_seq_occurrences, all_encountered_seq_occurrences)
    assert same(expected_all_encountered_context_occurrences, all_encountered_context_occurrences)
    assert same(expected_all_encountered_exploration_attempts, all_encountered_exploration_attempts)
    assert same(expected_all_encountered_exploration_results, all_encountered_exploration_results)
    assert same(expected_all_encountered_rewards_punishments, all_encountered_rewards_punishments)


def test_sort_all_encountered():
    flock_size = 2
    buffer_size = 5
    n_frequent_seqs = 3
    n_cluster_centers = 3
    context_size = 4
    n_providers = 1

    nan = FLOAT_NAN

    # WARN: Stick to this as cuda sorting is unstable
    device = 'cpu'
    float_dtype = get_float(device)

    process = create_tp_flock_learn_process(flock_size=flock_size, n_frequent_seqs=n_frequent_seqs,
                                            n_cluster_centers=n_cluster_centers,
                                            context_size=context_size, device=device)

    creator = AllocatingCreator(device)
    buffer = TPFlockBuffer(creator, flock_size, buffer_size, n_cluster_centers, n_frequent_seqs, context_size,
                           n_providers)

    buffer.seq_probs.stored_data = torch.tensor([[[0.4, 0.2, 0.4],
                                                  [0.9, 0.7, 0],
                                                  [0.1, 0.2, 0.7],
                                                  [0.4, 0.2, 0.4],
                                                  [nan, nan, nan]],
                                                 [[0.4, 0.3, 0.3],
                                                  [1, 0, 0],
                                                  [0.2, 0.7, 0.1],
                                                  [0.1, 0.1, 0.8],
                                                  [nan, nan, nan]]], dtype=float_dtype, device=device)

    all_encountered_seqs = torch.tensor([[[1, 0, 1],
                                          [2, 1, 0],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [0, 1, 0],
                                          [-1, -1, -1],
                                          [-1, -1, -1]],
                                         [[1, 0, 1],
                                          [0, 1, 2],
                                          [1, 2, 0],
                                          [-1, -1, -1],
                                          [1, 2, 1],
                                          [2, 0, 1],
                                          [-1, -1, -1]]], dtype=torch.int64, device=device)

    all_encountered_seq_occurrences = torch.tensor([[0.5, 0.5, 0, 0, 1.4, 0, 0],
                                                    [0.5, 0.5, 0.5, 0, 1, 0.6, 0]], dtype=float_dtype, device=device)

    # [2, 7, 3, 1, 2]
    all_encountered_context_occurrences = torch.tensor([[[[[2, 2]], [[2, 2]], [[2, 2]]],
                                                         [[[1, 1]], [[1, 1]], [[1, 1]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]]],
                                                        [[[[3, 3]], [[3, 3]], [[3, 3]]],
                                                         [[[4, 4]], [[4, 4]], [[4, 4]]],
                                                         [[[5, 5]], [[5, 5]], [[5, 5]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]]]], dtype=float_dtype,
                                                       device=device)

    all_encountered_exploration_attempts = torch.tensor([[[1],
                                                          [5.2],
                                                          [3],
                                                          [4],
                                                          [5],
                                                          [6],
                                                          [7]],
                                                         [[11],
                                                          [12],
                                                          [13],
                                                          [14],
                                                          [15],
                                                          [0],
                                                          [0]]], dtype=float_dtype)

    all_encountered_exploration_success_rates = torch.tensor([[[[3, 3, 3]],
                                                               [[4, 4, 4]],
                                                               [[5, 5, 5]],
                                                               [[6, 6, 6]],
                                                               [[7, 7, 7]],
                                                               [[8, 8, 8]],
                                                               [[9, 9, 9]]],
                                                              [[[0, 0, 0]],
                                                               [[1, 1, 1]],
                                                               [[2, 2, 2]],
                                                               [[3, 3, 3]],
                                                               [[4, 4, 4]],
                                                               [[5, 5, 5]],
                                                               [[6, 6, 6]]]], dtype=float_dtype)

    all_encountered_rewards_punishments = torch.tensor([[[[3, 3]],
                                                         [[4, 4]],
                                                         [[5, 5]],
                                                         [[6, 6]],
                                                         [[7, 7]],
                                                         [[8, 8]],
                                                         [[9, 9]]],
                                                        [[[0, 0]],
                                                         [[1, 1]],
                                                         [[2, 2]],
                                                         [[3, 3]],
                                                         [[4, 4]],
                                                         [[5, 5]],
                                                         [[6, 6]]]], dtype=float_dtype)

    process._sort_all_encountered(buffer,
                                  all_encountered_seq_occurrences,
                                  all_encountered_seqs,
                                  all_encountered_context_occurrences,
                                  all_encountered_rewards_punishments,
                                  all_encountered_exploration_attempts,
                                  all_encountered_exploration_success_rates)

    expectedstored_data = torch.tensor([[[nan, 0.4, 0.2],
                                         [nan, 0.9, 0.7],
                                         [nan, 0.1, 0.2],
                                         [nan, 0.4, 0.2],
                                         [nan, nan, nan]],
                                        [[nan, nan, 0.4],
                                         [nan, nan, 1],
                                         [nan, nan, 0.2],
                                         [nan, nan, 0.1],
                                         [nan, nan, nan]]], dtype=float_dtype, device=device)

    expected_all_encountered_seqs = torch.tensor([[[0, 1, 0],
                                                   [1, 0, 1],
                                                   [2, 1, 0],
                                                   [-1, -1, -1],
                                                   [-1, -1, -1],
                                                   [-1, -1, -1],
                                                   [-1, -1, -1]],
                                                  [[1, 2, 1],
                                                   [2, 0, 1],
                                                   [1, 0, 1],
                                                   [0, 1, 2],
                                                   [1, 2, 0],
                                                   [-1, -1, -1],
                                                   [-1, -1, -1]]], dtype=torch.int64, device=device)

    expected_all_encountered_seq_occurrences = torch.tensor([[1.4, 0.5, 0.5, 0, 0, 0, 0],
                                                             [1, 0.6, 0.5, 0.5, 0.5, 0, 0]], dtype=float_dtype,
                                                            device=device)

    expected_all_encountered_context_occurrences = torch.tensor([[[[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                                  [[[2, 2]], [[2, 2]], [[2, 2]]],
                                                                  [[[1, 1]], [[1, 1]], [[1, 1]]],
                                                                  [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                                  [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                                  [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                                  [[[0., 0.]], [[0., 0.]], [[0., 0.]]]],
                                                                 [[[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                                  [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                                  [[[3, 3]], [[3, 3]], [[3, 3]]],
                                                                  [[[4, 4]], [[4, 4]], [[4, 4]]],
                                                                  [[[5, 5]], [[5, 5]], [[5, 5]]],
                                                                  [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                                  [[[0., 0.]], [[0., 0.]], [[0., 0.]]]]],
                                                                dtype=float_dtype,
                                                                device=device)

    expected_all_encountered_exploration_attempts = torch.tensor([[[5],
                                                                   [1],
                                                                   [5.2],
                                                                   [3],
                                                                   [4],
                                                                   [6],
                                                                   [7]],
                                                                  [[15],
                                                                   [0],
                                                                   [11],
                                                                   [12],
                                                                   [13],
                                                                   [14],
                                                                   [0]]], dtype=float_dtype)

    expected_all_encountered_exploration_success_rates = torch.tensor([[[[7, 7, 7]],
                                                                        [[3, 3, 3]],
                                                                        [[4, 4, 4]],
                                                                        [[5, 5, 5]],
                                                                        [[6, 6, 6]],
                                                                        [[8, 8, 8]],
                                                                        [[9, 9, 9]]],
                                                                       [[[4, 4, 4]],
                                                                        [[5, 5, 5]],
                                                                        [[0, 0, 0]],
                                                                        [[1, 1, 1]],
                                                                        [[2, 2, 2]],
                                                                        [[3, 3, 3]],
                                                                        [[6, 6, 6]]]], dtype=float_dtype)

    expected_all_encountered_rewards_punishments = torch.tensor([[[[7, 7]],
                                                                  [[3, 3]],
                                                                  [[4, 4]],
                                                                  [[5, 5]],
                                                                  [[6, 6]],
                                                                  [[8, 8]],
                                                                  [[9, 9]]],
                                                                 [[[4, 4]],
                                                                  [[5, 5]],
                                                                  [[0, 0]],
                                                                  [[1, 1]],
                                                                  [[2, 2]],
                                                                  [[3, 3]],
                                                                  [[6, 6]]]], dtype=float_dtype)

    assert same(expectedstored_data, buffer.seq_probs.stored_data)
    assert same(expected_all_encountered_seqs, all_encountered_seqs)
    assert same(expected_all_encountered_seq_occurrences, all_encountered_seq_occurrences)
    assert same(expected_all_encountered_context_occurrences, all_encountered_context_occurrences)
    assert same(expected_all_encountered_exploration_attempts, all_encountered_exploration_attempts)
    assert same(expected_all_encountered_exploration_success_rates, all_encountered_exploration_success_rates)
    assert same(expected_all_encountered_rewards_punishments, all_encountered_rewards_punishments)


def test_forget():
    device = 'cpu'
    float_dtype = get_float(device)

    process = create_tp_flock_learn_process(device=device, forgetting_limit=3)

    all_encountered_seq_occurrences = torch.tensor([[0.5, 0.5, 0, 0, 1.4, 0, 0],
                                                    [1, 2, 1, 0, 1.5, 0.5, 0]], dtype=float_dtype)

    all_encountered_context_occurrences = torch.tensor([[[[0., 0.], [0., 0.], [0., 0.]],
                                                         [[2, 2], [2, 2], [2, 2]],
                                                         [[1, 1], [1, 1], [1, 1]],
                                                         [[0., 0.], [0., 0.], [0., 0.]],
                                                         [[0., 0.], [0., 0.], [0., 0.]],
                                                         [[0., 0.], [0., 0.], [0., 0.]],
                                                         [[0., 0.], [0., 0.], [0., 0.]]],
                                                        [[[0., 0.], [0., 0.], [0., 0.]],
                                                         [[0., 0.], [0., 0.], [0., 0.]],
                                                         [[3, 3], [3, 3], [3, 3]],
                                                         [[4, 4], [4, 4], [4, 4]],
                                                         [[5, 5], [5, 5], [5, 5]],
                                                         [[0., 0.], [0., 0.], [0., 0.]],
                                                         [[0., 0.], [0., 0.], [0., 0.]]]], dtype=float_dtype)

    all_encountered_exploration_attempts = torch.tensor([[[1], [2], [3], [7], [3], [3], [22]],
                                                         [[1], [2], [3], [10], [3], [11], [14]]], dtype=float_dtype)

    all_encountered_rewards_punishments = torch.tensor(
        [[[[1, 9]], [[2, 8]], [[3, 7]], [[4, 6]], [[5, 5]], [[6, 4]], [[7, 3]]],
         [[[1, 1]], [[2, 8]], [[3, 3]], [[4, 4]], [[5, 5]], [[6, 6]], [[7, 7]]]], dtype=float_dtype)

    process.total_encountered_occurrences.fill_(-1)

    process._forget(all_encountered_seq_occurrences, all_encountered_context_occurrences,
                    all_encountered_rewards_punishments, all_encountered_exploration_attempts,
                    process.total_encountered_occurrences)

    expected_all_encountered_seq_occurrences = torch.tensor([[0.5, 0.5, 0, 0, 1.4, 0, 0],
                                                             [0.5, 1, 0.5, 0, 0.75, 0.25, 0]], dtype=float_dtype)

    expected_total_encountered_occurrences = torch.tensor([2.4, 3], dtype=float_dtype)

    expected_all_encountered_context_occurrences = torch.tensor([[[[0., 0.], [0., 0.], [0., 0.]],
                                                                  [[2, 2], [2, 2], [2, 2]],
                                                                  [[1, 1], [1, 1], [1, 1]],
                                                                  [[0., 0.], [0., 0.], [0., 0.]],
                                                                  [[0., 0.], [0., 0.], [0., 0.]],
                                                                  [[0., 0.], [0., 0.], [0., 0.]],
                                                                  [[0., 0.], [0., 0.], [0., 0.]]],
                                                                 [[[0., 0.], [0., 0.], [0., 0.]],
                                                                  [[0., 0.], [0., 0.], [0., 0.]],
                                                                  [[1.5, 1.5], [1.5, 1.5], [1.5, 1.5]],
                                                                  [[2, 2], [2, 2], [2, 2]],
                                                                  [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]],
                                                                  [[0., 0.], [0., 0.], [0., 0.]],
                                                                  [[0., 0.], [0., 0.], [0., 0.]]]], dtype=float_dtype)

    # The lower limit for exploration attempts is (by default) the prior of 5.0
    expected_all_encountered_exploration_attempts = torch.tensor([[[5], [5], [5], [7], [5], [5], [22]],
                                                                  [[5], [5], [5], [5], [5], [5.5], [7]]],
                                                                 dtype=float_dtype)

    expected_all_encountered_rewards_punishments = torch.tensor([[[[1, 9]], [[2, 8]], [[3, 7]],
                                                                  [[4, 6]], [[5, 5]], [[6, 4]], [[7, 3]]],
                                                                 [[[0.5, 0.5]], [[1, 4]], [[1.5, 1.5]], [[2, 2]],
                                                                  [[2.5, 2.5]], [[3, 3]], [[3.5, 3.5]]]],
                                                                dtype=float_dtype)

    assert same(expected_all_encountered_seq_occurrences, all_encountered_seq_occurrences)
    assert same(expected_total_encountered_occurrences, process.total_encountered_occurrences)
    assert same(expected_all_encountered_context_occurrences, all_encountered_context_occurrences)
    assert same(expected_all_encountered_exploration_attempts, all_encountered_exploration_attempts)
    assert same(expected_all_encountered_rewards_punishments, all_encountered_rewards_punishments)


def test_extract_frequent_seqs():
    device = 'cpu'
    float_dtype = get_float(device)
    flock_size = 2
    context_size = 2
    n_providers = 1
    n_frequent_seqs = 3
    seq_len = 3
    n_cluster_centers = 3
    seq_lookahead = 1

    process = create_tp_flock_learn_process(device=device, flock_size=flock_size, context_size=context_size,
                                            n_providers=n_providers, n_frequent_seqs=n_frequent_seqs,
                                            seq_length=seq_len, n_cluster_centers=n_cluster_centers,
                                            seq_lookahead=seq_lookahead)

    all_encountered_seqs = torch.tensor([[[0, 1, 0],
                                          [1, 0, 1],
                                          [2, 1, 0],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1],
                                          [-1, -1, -1]],
                                         [[1, 2, 1],
                                          [2, 0, 1],
                                          [1, 0, 1],
                                          [0, 1, 2],
                                          [1, 2, 0],
                                          [-1, -1, -1],
                                          [-1, -1, -1]]], dtype=torch.int64)

    all_encountered_seq_occurrences = torch.tensor([[4.4, 3.5, 3.5, 0, 0, 0, 0],
                                                    [4, 3.6, 3.5, 3.5, 3.5, 0, 0]], dtype=float_dtype)

    all_encountered_context_occurrences = torch.tensor([[[[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[2, 2]], [[2, 2]], [[2, 2]]],
                                                         [[[1, 1]], [[1, 1]], [[1, 1]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]]],
                                                        [[[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[1.5, 1.5]], [[1.5, 1.5]], [[1.5, 1.5]]],
                                                         [[[2, 2]], [[2, 2]], [[2, 2]]],
                                                         [[[2.5, 2.5]], [[2.5, 2.5]], [[2.5, 2.5]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]],
                                                         [[[0., 0.]], [[0., 0.]], [[0., 0.]]]]], dtype=float_dtype)

    all_encountered_reward_punishments = torch.tensor([[[[0, 1]],
                                                        [[1, 0]],
                                                        [[2, 0.6]],
                                                        [[-1, -1]],
                                                        [[-1, -1]],
                                                        [[-1, -1]],
                                                        [[-1, -1]]],
                                                       [[[1, 2]],
                                                        [[2, 0]],
                                                        [[1, 0]],
                                                        [[0, 1]],
                                                        [[1, 2]],
                                                        [[-1, -1]],
                                                        [[-1, -1]]]], dtype=float_dtype)

    all_encountered_exploration_attempts = torch.tensor([[[1], [2], [4], [5], [7], [8], [9]],
                                                         [[5], [6], [7], [8], [9], [10], [11]]], dtype=float_dtype)

    all_encountered_exploration_results = torch.tensor([[[[0.1, 0.4, 0.3]],
                                                         [[0.2, 0.4, 0.5]],
                                                         [[0.4, 0.5, 0.1]],
                                                         [[0.4, 0.6, 0.7]],
                                                         [[0.2, 0.2, 0.1]],
                                                         [[0.1, 0.7, 0.3]],
                                                         [[0.5, 0.4, 0.3]]],
                                                        [[[0.5, 0.8, 1.1]],
                                                         [[0.6, 0.4, 3.2]],
                                                         [[0.7, 1.2, 0.6]],
                                                         [[0.9, 0.2, 0.6]],
                                                         [[0.3, 0.9, 0.0]],
                                                         [[0.5, 0.3, 0.2]],
                                                         [[0.4, 8.8, 9.4]]]],
                                                       dtype=float_dtype)

    frequent_seqs = torch.full((flock_size, n_frequent_seqs, seq_len), fill_value=-1, dtype=torch.int64)

    frequent_seq_occurrences = torch.full((flock_size, n_frequent_seqs), fill_value=-1, dtype=float_dtype)

    frequent_context_likelihoods = torch.full((flock_size, n_frequent_seqs, seq_len, n_providers, context_size),
                                              fill_value=-1, dtype=float_dtype)

    frequent_exploration_attempts = torch.full((flock_size, n_frequent_seqs, seq_lookahead), fill_value=-1,
                                               dtype=float_dtype)
    frequent_exploration_results = torch.full((flock_size, n_frequent_seqs, seq_lookahead, n_cluster_centers),
                                              fill_value=-1, dtype=float_dtype)

    frequent_rewards_punishments = torch.full((flock_size, n_frequent_seqs, seq_lookahead, 2), fill_value=-1,
                                              dtype=float_dtype)

    process._extract_frequent_seqs(all_encountered_seqs,
                                   all_encountered_seq_occurrences,
                                   all_encountered_context_occurrences,
                                   all_encountered_reward_punishments,
                                   all_encountered_exploration_attempts,
                                   all_encountered_exploration_results,
                                   frequent_seqs,
                                   frequent_seq_occurrences,
                                   frequent_context_likelihoods,
                                   frequent_rewards_punishments,
                                   frequent_exploration_attempts,
                                   frequent_exploration_results)

    expected_frequent_seqs = torch.tensor([[[0, 1, 0],
                                            [1, 0, 1],
                                            [2, 1, 0]],
                                           [[1, 2, 1],
                                            [2, 0, 1],
                                            [1, 0, 1]]], dtype=torch.int64)

    expected_frequent_seq_occurrences = torch.tensor([[4.4, 3.5, 3.5],
                                                      [4, 3.6, 3.5]], dtype=float_dtype)

    expected_frequent_context_likelihoods = torch.tensor([[[[[0.3472, 0.3472]],
                                                            [[0.3472, 0.3472]],
                                                            [[0.3472, 0.3472]]],

                                                           [[[0.5185, 0.5185]],
                                                            [[0.5185, 0.5185]],
                                                            [[0.5185, 0.5185]]],

                                                           [[[0.4444, 0.4444]],
                                                            [[0.4444, 0.4444]],
                                                            [[0.4444, 0.4444]]]],

                                                          [[[[0.3571, 0.3571]],
                                                            [[0.3571, 0.3571]],
                                                            [[0.3571, 0.3571]]],

                                                           [[[0.3676, 0.3676]],
                                                            [[0.3676, 0.3676]],
                                                            [[0.3676, 0.3676]]],

                                                           [[[0.4815, 0.4815]],
                                                            [[0.4815, 0.4815]],
                                                            [[0.4815, 0.4815]]]]],
                                                         dtype=float_dtype)

    expected_frequent_exploration_attempts = torch.tensor([[[1], [2], [4]],
                                                           [[5], [6], [7]]], dtype=float_dtype)

    expected_frequent_exploration_results = torch.tensor([[[[0.1, 0.4, 0.3]],
                                                           [[0.2, 0.4, 0.5]],
                                                           [[0.4, 0.5, 0.1]]],
                                                          [[[0.5, 0.8, 1.1]],
                                                           [[0.6, 0.4, 3.2]],
                                                           [[0.7, 1.2, 0.6]]]], dtype=float_dtype)

    expected_frequent_reward_punishments = torch.tensor([[[[0, 1]],
                                                          [[1, 0]],
                                                          [[2, 0.6]]],
                                                         [[[1, 2]],
                                                          [[2, 0]],
                                                          [[1, 0]]]], dtype=float_dtype)

    assert same(expected_frequent_seqs, frequent_seqs)
    assert same(expected_frequent_seq_occurrences, frequent_seq_occurrences)
    assert same(expected_frequent_context_likelihoods, frequent_context_likelihoods, eps=1e-4)
    assert same(expected_frequent_exploration_attempts, frequent_exploration_attempts)
    assert same(expected_frequent_exploration_results, frequent_exploration_results)
    assert same(expected_frequent_reward_punishments, frequent_rewards_punishments)


@pytest.mark.parametrize("flock_size, batch_size, n_subbatches",
                         [(2, 25, 2), (1, 5, 1), (10, 22, 7), (3, 12, 12)])
def test_subbatch(flock_size, batch_size, n_subbatches):
    device = 'cpu'
    float_dtype = get_float(device)
    n_cluster_centers = 3

    process = create_tp_flock_learn_process(device=device, flock_size=flock_size,
                                            batch_size=batch_size, n_subbatches=n_subbatches)

    clusters_tensor = torch.rand((flock_size, batch_size, n_cluster_centers), dtype=float_dtype, device=device)

    subbatches_capacity = n_subbatches * process._subbatch_size - (n_subbatches - 1) * process._subbatch_overlap
    padding_length = subbatches_capacity - process._combined_batch_size

    process._subbatch(clusters_tensor, process.cluster_subbatch, padding_length)

    expected_padding = torch.zeros(flock_size, padding_length, n_cluster_centers)

    # Test that the padding (if there is any) is where we expect it to be
    if padding_length > 0:
        assert same(expected_padding,
                    process.cluster_subbatch.view(flock_size, -1, n_cluster_centers)[:, -padding_length:])

    # Test that the overlap region at the end of subbatch k is the same as the region at the beginning of k+1
    for k in range(n_subbatches - 1):
        assert (same(process.cluster_subbatch[:, k, -process._subbatch_overlap:],
                     process.cluster_subbatch[:, k + 1, :process._subbatch_overlap]))
