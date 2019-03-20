import pytest

import torch
from torchsim.core import get_float, SMALL_CONSTANT
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.expert_params import DEFAULT_CONTEXT_PRIOR, DEFAULT_EXPLORATION_ATTEMPTS_PRIOR, ExpertParams
from torchsim.core.models.temporal_pooler import TPFlockBuffer, ConvTPFlockLearning, ConvTPFlock
from torchsim.core.utils.tensor_utils import same


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
                                  flock_size=2,
                                  max_encountered_seqs=7,
                                  max_new_seqs=3,
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
                                  n_providers=1,
                                  device='cuda'):
    float_dtype = get_float(device)
    seq_lookbehind = seq_length - seq_lookahead

    all_indices = torch.arange(end=flock_size, device=device).unsqueeze(dim=1)

    if buffer is None:
        buffer = create_tp_buffer(flock_size=flock_size, device=device)

    if frequent_seqs is None:
        frequent_seqs = torch.full((1, n_frequent_seqs, seq_length), fill_value=-1.,
                                   dtype=torch.int64, device=device)

    if frequent_seq_occurrences is None:
        frequent_seq_occurrences = torch.full((1, n_frequent_seqs), fill_value=-1., dtype=float_dtype,
                                              device=device)

    if frequent_context_likelihoods is None:
        frequent_context_likelihoods = torch.full((1, n_frequent_seqs, seq_length, n_providers, context_size),
                                               fill_value=-1., dtype=float_dtype, device=device)

    if frequent_exploration_attempts is None:
        frequent_exploration_attempts = torch.full((flock_size, n_frequent_seqs, seq_lookahead, n_cluster_centers),
                                                   fill_value=-1., dtype=float_dtype, device=device)

    if frequent_exploration_results is None:
        frequent_exploration_results = torch.full((flock_size, n_frequent_seqs, seq_lookahead, n_cluster_centers),
                                                        fill_value=-1., dtype=float_dtype, device=device)

    if all_encountered_exploration_attempts is None:
        all_encountered_exploration_attempts = torch.full((flock_size, max_encountered_seqs, seq_lookahead, n_cluster_centers),
                                                          fill_value=-1., dtype=float_dtype, device=device)

    if all_encountered_exploration_success_rates is None:
        all_encountered_exploration_success_rates = torch.full((flock_size, max_encountered_seqs, seq_lookahead, n_cluster_centers),
                                                               fill_value=-1., dtype=float_dtype, device=device)

    if execution_counter_learning is None:
        execution_counter_learning = torch.zeros((flock_size, 1), device=device, dtype=float_dtype)

    if all_encountered_seqs is None:
        all_encountered_seqs = torch.full((1, max_encountered_seqs, seq_length), fill_value=-1,
                                          dtype=torch.int64, device=device)

    if all_encountered_seq_occurrences is None:
        all_encountered_seq_occurrences = torch.zeros(1, max_encountered_seqs, dtype=float_dtype, device=device)

    if all_encountered_context_occurrences is None:
        all_encountered_context_occurrences = torch.full((1, max_encountered_seqs, seq_length, n_providers, context_size),
                                                         fill_value=-1, dtype=float_dtype, device=device)

    if all_encountered_rewards_punishments is None:
        all_encountered_rewards_punishments = torch.zeros((1, max_encountered_seqs, seq_lookahead, 2),
                                                          dtype=float_dtype, device=device)

    if frequent_rewards_punishments is None:
        frequent_rewards_punishments = torch.zeros((1, n_frequent_seqs, seq_lookahead, 2), dtype=float_dtype, device=device)

    return ConvTPFlockLearning(all_indices,
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


def test_update_knowledge_new_seqs():
    """
    Unlike the non-conv TP, there is only one sets of new sequences, and one set of all encountered sequences
    irrespective of the number of experts that are learning in this process.
    """

    flock_size = 2
    n_providers = 1
    max_encountered_seqs = 7
    n_frequent_seqs = 3
    seq_length = 3
    n_cluster_centers = 3
    batch_size = 3
    forgetting_limit = 3
    context_size = 4
    seq_lookahead = 1
    max_new_seqs = 3

    device = 'cpu'
    dtype = get_float(device)

    learning_proc = create_tp_flock_learn_process(flock_size=flock_size, max_encountered_seqs=max_encountered_seqs,
                                                  n_frequent_seqs=n_frequent_seqs, seq_length=seq_length,
                                                  n_cluster_centers=n_cluster_centers, batch_size=batch_size,
                                                  forgetting_limit=forgetting_limit, context_size=context_size,
                                                  device=device, seq_lookahead=seq_lookahead, n_providers=n_providers)

    learning_proc._all_encountered_seqs[0, 0] = torch.tensor([1, 2, 3], device=device, dtype=torch.int64)
    learning_proc._all_encountered_seq_occurrences[0, 0] = 3

    learning_proc._all_encountered_seqs[0, 1] = torch.tensor([2, 1, 0], device=device, dtype=torch.int64)
    learning_proc._all_encountered_seq_occurrences[0, 1] = 0.4

    all_enc_seqs = learning_proc._all_encountered_seqs
    all_enc_seq_occ = learning_proc._all_encountered_seq_occurrences

    all_enc_cont_occ = torch.full((1, max_encountered_seqs, seq_length, n_providers, context_size),
                                  fill_value=-1, dtype=dtype, device=device)
    all_enc_exp_att = torch.full((flock_size, max_encountered_seqs, seq_lookahead, n_cluster_centers),
                                 fill_value=-1., dtype=dtype, device=device)
    all_enc_exp_suc_rates = torch.full((flock_size, max_encountered_seqs, seq_lookahead, n_cluster_centers),
                                       fill_value=-1., dtype=dtype, device=device)
    all_enc_rew_pun = torch.full((flock_size, max_encountered_seqs, seq_lookahead, 2), fill_value=-1,
                                 dtype=dtype, device=device)

    most_probable_batch_seqs = torch.tensor([[[0, 1, 3], [2, 1, 3], [-1, -1, -1], [3, 1, 3]]],
                                            device=device, dtype=torch.int64)
    newly_enc_seq_counts = torch.tensor([[2, 3, 0, 1]], device=device, dtype=dtype)

    learning_proc._update_knowledge_new_seqs(all_enc_seqs, all_enc_seq_occ, all_enc_cont_occ, all_enc_rew_pun, all_enc_exp_att,
                                             all_enc_exp_suc_rates, most_probable_batch_seqs, newly_enc_seq_counts)

    expected_all_enc_seqs = all_enc_seqs.clone()
    expected_all_enc_seqs[0, -max_new_seqs:] = torch.tensor([[2, 1, 3], [0, 1, 3], [3, 1, 3]], dtype=torch.int64)

    expected_all_enc_seq_occ = all_enc_seq_occ.clone()
    expected_all_enc_seq_occ[0, -max_new_seqs:] = torch.tensor([3, 2, 1], dtype=dtype)

    new_context_counts = expected_all_enc_seq_occ[:, -max_new_seqs:].view(1, max_new_seqs, 1, 1, 1).expand(1, max_new_seqs, seq_length, n_providers, context_size) / 2

    expected_all_enc_cont_occ = all_enc_cont_occ.clone()
    expected_all_enc_cont_occ[:, -max_new_seqs:, :, :, :] = new_context_counts

    expected_all_enc_exp_att = all_enc_exp_att.clone()
    expected_all_enc_exp_att[:, -max_new_seqs:, :, :] = learning_proc._exploration_attempts_prior

    expected_all_enc_exp_suc_rates = all_enc_exp_suc_rates.clone()
    expected_all_enc_exp_suc_rates[:, -max_new_seqs:, :, :] = 0

    expected_all_enc_rew_pun = all_enc_rew_pun.clone()
    expected_all_enc_rew_pun[: -max_new_seqs:, :, :] = 0

    assert same(expected_all_enc_seqs, all_enc_seqs)
    assert same(expected_all_enc_seq_occ, all_enc_seq_occ)
    assert same(expected_all_enc_cont_occ, all_enc_cont_occ)
    assert same(expected_all_enc_exp_att, all_enc_exp_att)
    assert same(expected_all_enc_exp_suc_rates, all_enc_exp_suc_rates)
    assert same(expected_all_enc_rew_pun, all_enc_rew_pun)


def test_forward_learn_disambiguating_context():
    """Tests if context helps to disambiguate situations.

        Two experts, both seeing the same data and contexts
    """
    device = 'cuda'
    float_dtype = get_float(device)
    batch_size = 5

    params = ExpertParams()
    params.flock_size = 2
    params.n_cluster_centers = 3

    tp_params = params.temporal
    tp_params.n_providers = 1
    tp_params.incoming_context_size = 2
    tp_params.seq_length = 2
    tp_params.buffer_size = 1000
    tp_params.batch_size = batch_size
    tp_params.seq_lookahead = 1
    tp_params.n_frequent_seqs = 50
    tp_params.max_encountered_seqs = 300
    tp_params.forgetting_limit = 1
    tp_params.learning_period = 1
    tp_params.context_prior = SMALL_CONSTANT
    tp_params.exploration_probability = 0
    tp_params.follow_goals = False
    tp_params.enable_learning = True
    tp_params.max_new_seqs = 4
    tp_params.compute_backward_pass = True

    flock = ConvTPFlock(params, creator=AllocatingCreator(device))

    # experts:
    #   both receive a cycling sequence and useful context

    seqs = torch.tensor([[[1, 0, 0], [1, 0, 0]],
                         [[0, 1, 0], [0, 1, 0]],
                         [[0, 0, 1], [0, 0, 1]],
                         [[0, 1, 0], [0, 1, 0]]], dtype=float_dtype, device=device)

    contexts = torch.tensor([[[[[1, 0], [0, 0], [0, 0]]], [[[1, 0], [0, 0], [0, 0]]]],
                             [[[[1, 0], [0, 0], [0, 0]]], [[[1, 0], [0, 0], [0, 0]]]],
                             [[[[1, 0], [0, 0], [0, 0]]], [[[1, 0], [0, 0], [0, 0]]]],
                             [[[[0, 1], [0, 0], [0, 0]]], [[[0, 1], [0, 0], [0, 0]]]]], dtype=float_dtype, device=device)

    expected_actions = torch.tensor([[[0, 1, 0], [0, 1, 0]],
                                     [[0, 0, 1], [0, 0, 1]],
                                     [[0, 1, 0], [0, 1, 0]],
                                     [[1, 0, 0], [1, 0, 0]]], dtype=float_dtype, device=device)

    expected_projections = torch.tensor([[[0.6666, 0.3333, 0.], [0.6666, 0.3333, 0.]],
                                         [[0., 0.6666, 0.3333], [0., 0.6666, 0.3333]],
                                         [[0., 0.3333, 0.6666], [0., 0.3333, 0.6666]],
                                         [[0.3333, 0.6666, 0.], [0.3333, 0.6666, 0.]]], dtype=float_dtype,
                                        device=device)

    # need to learn at least 2 times because contexts are not extracted when sequence is new. And they are regarded
    #  as uniform context so we need even more to get to a sharp probability
    iterations = batch_size * 4
    for k in range(iterations):
        cluster_data = seqs[k % 4]
        context_data = contexts[k % 4]
        flock.forward_learn(cluster_data, context_data, input_rewards=None)

    # Run through the sequences yet again and no grab the outputs each time for flock 2
    for k in range(4):
        flock._forward(seqs[k], contexts[k], input_rewards=None, sp_mask=None)
        # Check that the predicted outputs are what we expect
        assert same(expected_actions[k], flock.action_outputs, eps=1e-1)
        # Check that the output projection is what we expect
        assert same(expected_projections[k], flock.projection_outputs, eps=3e-2)


@pytest.mark.parametrize("flock_size, batch_size, n_cluster_centers, overlap", [(2, 5, 3, 2), (100, 60, 17, 4),
                                                                                (22, 800, 4, 60)])
def test_pad_and_combine_tensor(flock_size, batch_size, n_cluster_centers, overlap):
    max_encountered_seqs = 7
    n_frequent_seqs = 3
    seq_length = 3
    forgetting_limit = 3
    context_size = 4
    seq_lookahead = 1

    device = 'cpu'
    dtype = get_float(device)

    learning_proc = create_tp_flock_learn_process(flock_size=flock_size, max_encountered_seqs=max_encountered_seqs,
                                                  n_frequent_seqs=n_frequent_seqs, seq_length=seq_length,
                                                  n_cluster_centers=n_cluster_centers, batch_size=batch_size,
                                                  forgetting_limit=forgetting_limit, context_size=context_size,
                                                  device=device, seq_lookahead=seq_lookahead)

    tensor = torch.arange(0, flock_size * batch_size,
                          dtype=dtype, device=device).unsqueeze(1).expand(flock_size * batch_size,
                                                                          n_cluster_centers).view(flock_size,
                                                                                                  batch_size,
                                                                                                  n_cluster_centers)

    learning_proc._subbatch_overlap = overlap
    combined_tensor = learning_proc.pad_and_combine_tensor(tensor)

    expected_tensor = torch.cat([tensor, torch.zeros((flock_size, overlap, n_cluster_centers), dtype=dtype, device=device)], dim=1)
    expected_tensor = expected_tensor.view(1, -1, n_cluster_centers)

    assert same(expected_tensor, combined_tensor)


def test_convolutional_aspect():
    """Tests whether the convTP is actually convolutional,

    The convolutional TP should allow any expert to recognise and predict the next element in a sequence when one
    expert does. This test shows two experts two separate sequences and, then tests them using the sequence which the
    other expert trained.

    Because lookbehind is 2, before testing, the experts need to be 'primed' with the cluster immediately preceeding
    the sequence they are to be tested on.
    """

    device = 'cuda'
    float_dtype = get_float(device)
    batch_size = 10

    params = ExpertParams()
    params.flock_size = 2
    params.n_cluster_centers = 6

    tp_params = params.temporal
    tp_params.incoming_context_size = 2
    tp_params.seq_length = 3
    tp_params.buffer_size = 1000
    tp_params.batch_size = batch_size
    tp_params.seq_lookahead = 1
    tp_params.n_frequent_seqs = 50
    tp_params.max_encountered_seqs = 300
    tp_params.forgetting_limit = 1
    tp_params.learning_period = 1
    tp_params.context_prior = SMALL_CONSTANT
    tp_params.exploration_probability = 0
    tp_params.follow_goals = False
    tp_params.enable_learning = True
    tp_params.max_new_seqs = 4
    tp_params.compute_backward_pass = True

    flock = ConvTPFlock(params, creator=AllocatingCreator(device))

    training_seqs = torch.tensor([[[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]],
                                  [[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]],
                                  [[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]]], dtype=float_dtype, device=device)

    training_contexts = torch.tensor([[[[[1, 0], [0, 0], [0, 0]]], [[[0, 1], [0, 0], [0, 0]]]],
                                      [[[[1, 0], [0, 0], [0, 0]]], [[[0, 1], [0, 0], [0, 0]]]],
                                      [[[[1, 0], [0, 0], [0, 0]]], [[[0, 1], [0, 0], [0, 0]]]],
                                      [[[[1, 0], [0, 0], [0, 0]]], [[[0, 1], [0, 0], [0, 0]]]]], dtype=float_dtype, device=device)

    expected_actions_1 = torch.tensor([[[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]],
                                       [[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]],
                                       [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]], dtype=float_dtype, device=device)

    # Train the models
    iterations = batch_size + 10
    for k in range(iterations):
        for cluster_data, context_data in zip(training_seqs, training_contexts):
            flock.forward_learn(cluster_data, context_data, input_rewards=None)

    # Test that each expert can recall the sequences that it had learned
    for cluster_data, context_data, expected_action in zip(training_seqs, training_contexts, expected_actions_1):
        flock._forward(cluster_data, context_data, input_rewards=None, sp_mask=None)
        # Check that the predicted outputs are what we expect
        assert same(expected_action, flock.action_outputs, eps=1e-1)

    # There is a probably a cool way to reshuffle the innermost values of the tensor, but for now,
    # just define what we want to test with manually.

    testing_seqs = torch.tensor([[[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]],
                                 [[0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0]],
                                 [[0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0]]], dtype=float_dtype, device=device)

    testing_contexts = torch.tensor([[[[[0, 1], [0, 0], [0, 0]]], [[[1, 0], [0, 0], [0, 0]]]],
                                     [[[[0, 1], [0, 0], [0, 0]]], [[[1, 0], [0, 0], [0, 0]]]],
                                     [[[[0, 1], [0, 0], [0, 0]]], [[[1, 0], [0, 0], [0, 0]]]],
                                     [[[[0, 1], [0, 0], [0, 0]]], [[[1, 0], [0, 0], [0, 0]]]]], dtype=float_dtype, device=device)

    expected_actions = torch.tensor([[[0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0]],
                                     [[0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0]],
                                     [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]]], dtype=float_dtype, device=device)

    # Because each expert maintains its own buffer, we need to "prime" them with a single example from the sequence we
    # want them to now recognise. This is the last part of testing seqs, so it lines up with the first example next time
    flock._forward(testing_seqs[2], testing_contexts[2], input_rewards=None, sp_mask=None)

    # Test that each expert can recall the sequences that *the other expert* had learned
    for cluster_data, context_data, expected_action in zip(testing_seqs, testing_contexts, expected_actions):
        flock._forward(cluster_data, context_data, input_rewards=None, sp_mask=None)
        # Check that the predicted outputs are what we expect
        assert same(expected_action, flock.action_outputs, eps=1e-1)


@pytest.mark.parametrize("flock_size, batch_size, overlap", [(3, 20, 3), (5, 10, 2)])
def test_combine_flocks(flock_size, batch_size, overlap):
    """Test the mechanism for combining all the batches from each flock into 4 tensors with padding in between.

    """
    max_encountered_seqs = 7
    n_frequent_seqs = 3
    seq_length = 3
    n_cluster_centers = 3
    forgetting_limit = 3
    context_size = 4
    seq_lookahead = 1

    device = 'cpu'
    dtype = get_float(device)

    learning_proc = create_tp_flock_learn_process(flock_size=flock_size, max_encountered_seqs=max_encountered_seqs,
                                                  n_frequent_seqs=n_frequent_seqs, seq_length=seq_length,
                                                  n_cluster_centers=n_cluster_centers, batch_size=batch_size,
                                                  forgetting_limit=forgetting_limit, context_size=context_size,
                                                  device=device, seq_lookahead=seq_lookahead)

    cluster_batch = torch.rand((flock_size, batch_size, n_cluster_centers), dtype=dtype, device=device)
    context_batch = torch.rand((flock_size, batch_size, context_size), dtype=dtype, device=device)
    exploring_batch = torch.rand((flock_size, batch_size, 1), dtype=dtype, device=device)
    actions_batch = torch.rand((flock_size, batch_size, n_cluster_centers), dtype=dtype, device=device)
    rewards_batch = torch.rand((flock_size, batch_size, 2), dtype=dtype, device=device)

    learning_proc._subbatch_overlap = overlap

    comb_clusters, comb_contexts, comb_rewards, comb_exploring, comb_actions = learning_proc._combine_flocks(cluster_batch,
                                                                                               context_batch,
                                                                                               rewards_batch,
                                                                                               exploring_batch,
                                                                                               actions_batch)

    expected_comb_clusters = torch.cat([cluster_batch, torch.zeros(flock_size, overlap, n_cluster_centers)], dim=1).view(1, -1, n_cluster_centers)
    expected_comb_contexts = torch.cat([context_batch, torch.zeros(flock_size, overlap, context_size)], dim=1).view(1, -1, context_size)
    expected_comb_exploring = torch.cat([exploring_batch, torch.zeros(flock_size, overlap, 1)], dim=1).view(1, -1, 1)
    expected_comb_actions = torch.cat([actions_batch, torch.zeros(flock_size, overlap, n_cluster_centers)], dim=1).view(1, -1, n_cluster_centers)
    expected_comb_rewards = torch.cat([rewards_batch, torch.zeros(flock_size, overlap, 2)], dim=1).view(1, -1, 2)

    expected_valid_seqs_indicator = torch.cat([torch.ones((flock_size, batch_size-overlap), dtype=torch.int64),
                                               torch.zeros((flock_size, overlap * 2), dtype=torch.int64)], dim=1).view(1, -1)[:, : learning_proc._max_seqs_in_batch]

    assert same(expected_comb_clusters, comb_clusters, eps=1e-3)
    assert same(expected_comb_contexts, comb_contexts, eps=1e-3)
    assert same(expected_comb_exploring, comb_exploring, eps=1e-3)
    assert same(expected_comb_actions, comb_actions, eps=1e-3)
    assert same(expected_comb_rewards, comb_rewards, eps=1e-3)

    assert same(expected_valid_seqs_indicator, learning_proc._combined_valid_seqs)
