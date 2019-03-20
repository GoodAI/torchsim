import torch

from torchsim.core import get_float, SMALL_CONSTANT, FLOAT_NAN
from torchsim.core.models.temporal_pooler import TPFlockUntrainedForwardAndBackward
from torchsim.core.utils.tensor_utils import normalize_probs_, move_probs_towards_50_, add_small_constant_, same
from tests.core.models.temporal_pooler.test_forward_process import create_tp_buffer


def create_tp_flock_untrained_forward_process(all_indices=None,
                                              flock_size=2,
                                              n_frequent_seqs=4,
                                              n_cluster_centers=3,
                                              context_size=5,
                                              buffer=None,
                                              cluster_data=None,
                                              context_data=None,
                                              reward_data=None,
                                              projection_outputs=None,
                                              action_outputs=None,
                                              do_subflocking=True,
                                              device='cuda'):
    if all_indices is None:
        all_indices = torch.arange(end=flock_size, dtype=torch.int64, device=device).unsqueeze(dim=1)

    if buffer is None:
        buffer = create_tp_buffer(flock_size=flock_size, device=device)

    if cluster_data is None:
        cluster_data = torch.zeros((flock_size, n_cluster_centers), device=device)

    if context_data is None:
        context_data = torch.zeros((flock_size, 2, context_size), device=device)

    if reward_data is None:
        reward_data = torch.zeros((flock_size, 2), device=device)

    if projection_outputs is None:
        projection_outputs = torch.zeros((flock_size, n_cluster_centers), device=device)

    if action_outputs is None:
        action_outputs = torch.zeros((flock_size, n_cluster_centers), device=device)

    return TPFlockUntrainedForwardAndBackward(all_indices,
                                              do_subflocking,
                                              buffer,
                                              cluster_data,
                                              context_data,
                                              reward_data,
                                              projection_outputs,
                                              action_outputs,
                                              n_frequent_seqs,
                                              n_cluster_centers,
                                              device)


def test_forward_passive():
    device = 'cuda'
    float_dtype = get_float(device)
    flock_size = 2
    buffer_size = 6
    context_size = 2
    n_cluster_centers = 3
    n_frequent_seqs = 3
    n_providers = 1

    buffer = create_tp_buffer(flock_size=flock_size,
                              buffer_size=buffer_size,
                              n_cluster_centers=n_cluster_centers,
                              n_frequent_seqs=n_frequent_seqs,
                              context_size=context_size,
                              n_providers=n_providers,
                              device=device)

    # region setup tensors

    nan = FLOAT_NAN
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

    frequent_context_likelihoods = torch.tensor([[[[4.5, 2.5], [1, 2.5], [3, 2.5]],
                                                      [[4, 2], [0.1, 2], [3, 2]],
                                                      [[0, 0], [0, 0], [0, 0]]],
                                                     [[[0, 1.5], [1, 1.5], [2.9, 1.5]],
                                                      [[0, 0], [0, 0], [0, 0]],
                                                      [[0, 0], [0, 0], [0, 0]]]], dtype=float_dtype, device=device)
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

    # Pre-fill the output tensors so that we can check that they were written into
    projection_output = torch.full((flock_size, n_cluster_centers), fill_value=-2, dtype=float_dtype, device=device)
    action_output = torch.full((flock_size, n_cluster_centers), fill_value=-2, dtype=float_dtype,
                               device=device)

    # endregion

    process = create_tp_flock_untrained_forward_process(flock_size=flock_size,
                                                        n_frequent_seqs=n_frequent_seqs,
                                                        n_cluster_centers=n_cluster_centers,
                                                        context_size=context_size,
                                                        buffer=buffer,
                                                        cluster_data=cluster_data,
                                                        context_data=context_data,
                                                        projection_outputs=projection_output,
                                                        action_outputs=action_output,
                                                        do_subflocking=True,
                                                        device=device)

    process.run_and_integrate()

    # temporary process expected values

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
                                               [0, 0, 0],  # current_ptr
                                               [2, 2, 2],
                                               [2, 2, 2],
                                               [2, 2, 2]],
                                              [[2, 2, 2],
                                               [2, 2, 2],
                                               [2, 2, 2],
                                               [2, 2, 2],
                                               [0, 0, 0],  # current_ptr
                                               [2, 2, 2]]], dtype=float_dtype, device=device)

    fill_value = (1.0 / n_cluster_centers)
    # There are 3 cluster centers.
    expected_buffer_outputs = torch.tensor([[[2, 2, 2],
                                             [2, 2, 2],
                                             [fill_value, fill_value, fill_value],  # current_ptr
                                             [2, 2, 2],
                                             [2, 2, 2],
                                             [2, 2, 2]],
                                            [[2, 2, 2],
                                             [2, 2, 2],
                                             [2, 2, 2],
                                             [2, 2, 2],
                                             [fill_value, fill_value, fill_value],  # current_ptr
                                             [2, 2, 2]]], dtype=float_dtype, device=device)

    expected_projection_output = torch.full((2, 3), fill_value=fill_value, dtype=float_dtype, device=device)
    expected_action_output = torch.full((2, 3), fill_value=fill_value, dtype=float_dtype, device=device)

    # endregion

    assert same(expected_projection_output, projection_output, eps=1e-4)
    assert same(expected_action_output, action_output, eps=1e-4)
    # test also storing into buffer
    assert same(expected_buffer_current_ptr, buffer.current_ptr)
    assert same(expected_buffer_total_data_written, buffer.total_data_written)
    assert same(expected_buffer_outputs, buffer.outputs.stored_data, eps=1e-4)
    assert same(expected_buffer_seq_probs, buffer.seq_probs.stored_data, eps=1e-4)
    assert same(expected_buffer_clusters, buffer.clusters.stored_data, eps=1e-4)
    assert same(expected_buffer_contexts, buffer.contexts.stored_data, eps=1e-4)
