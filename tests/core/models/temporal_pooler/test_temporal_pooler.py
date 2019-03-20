from typing import List, Any
from unittest.mock import patch

import torch
import pytest

from torchsim.core import get_float, SMALL_CONSTANT
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.expert_params import ExpertParams, DEFAULT_CONTEXT_PRIOR, NUMBER_OF_CONTEXT_TYPES
from torchsim.core.models.temporal_pooler import TPFlock, UntrainedForwardProcessFactory, \
    TrainedForwardProcessFactory, TPOutputProjection
from torchsim.core.utils.tensor_utils import same
from tests.testing_utils import copy_or_fill_tensor


def create_tp_flock(n_cluster_centers=3, flock_size=1, seq_length=3, buffer_size=10, batch_size=5,
                    seq_lookahead=1, n_frequent_seq=4, max_encountered_seq=20,
                    forgetting_limit=40, learning_period=1, incoming_context_size=1,
                    context_prior=DEFAULT_CONTEXT_PRIOR, exploration_probability=0, follow_goals=False,
                    n_providers=1,
                    trained_forward_factory=None, untrained_forward_factory=None, enable_learning=True, device='cuda'):
    params = ExpertParams()
    params.flock_size = flock_size
    params.n_cluster_centers = n_cluster_centers
    params.flock_size = flock_size

    tp_params = params.temporal
    tp_params.incoming_context_size = incoming_context_size
    tp_params.seq_length = seq_length
    tp_params.buffer_size = buffer_size
    tp_params.batch_size = batch_size
    tp_params.seq_lookahead = seq_lookahead
    tp_params.n_frequent_seqs = n_frequent_seq
    tp_params.max_encountered_seqs = max_encountered_seq
    tp_params.forgetting_limit = forgetting_limit
    tp_params.learning_period = learning_period
    tp_params.context_prior = context_prior
    tp_params.exploration_probability = exploration_probability
    tp_params.follow_goals = follow_goals
    tp_params.enable_learning = enable_learning
    tp_params.n_providers = n_providers
    tp_params.frustration_threshold = 10000
    tp_params.follow_goals = False
    tp_params.compute_backward_pass = True

    return TPFlock(params, creator=AllocatingCreator(device),
                   trained_forward_factory=trained_forward_factory,
                   untrained_forward_factory=untrained_forward_factory)


# TODO (Test): add a test for follow_goals=True
@pytest.mark.slow
def test_forward_learn_no_context():
    device = 'cuda'
    float_dtype = get_float(device)
    flock = create_tp_flock(flock_size=2, seq_length=3, buffer_size=1000, batch_size=20, max_encountered_seq=800,
                            incoming_context_size=4, exploration_probability=0)

    # experts:
    #   ex1 receives the same input all the time - doesn't learn
    #   ex2 receives a cycling sequence - this one should learn appropriately

    seqs = torch.tensor([[[1, 0, 0], [1, 0, 0]],
                         [[1, 0, 0], [0, 1, 0]],
                         [[1, 0, 0], [0, 0, 1]],
                         [[1, 0, 0], [0, 1, 0]]], dtype=float_dtype, device=device)

    context = torch.tensor([[[[1, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]], [[[1, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]]], dtype=float_dtype,
                           device=device)

    # If the forward pass doesnt run, the outputs should always be zero for ex1
    expected_actions = torch.tensor([[[0., 0., 0.], [0, 1, 0]],
                                     [[0., 0., 0.], [0, 0, 1]],
                                     [[0., 0., 0.], [0, 1, 0]],
                                     [[0., 0., 0.], [1, 0, 0]]], dtype=float_dtype, device=device)

    expected_projections = torch.tensor([[[0.3333, 0.3333, 0.3333], [0.5, 0.5, 0]],
                                         [[0.3333, 0.3333, 0.3333], [0.25, 0.5, 0.25]],
                                         [[0.3333, 0.3333, 0.3333], [0, 0.5, 0.5]],
                                         [[0.3333, 0.3333, 0.3333], [0.25, 0.5, 0.25]]], dtype=float_dtype,
                                        device=device)

    expected_buffer_total_data_written = torch.tensor([1, 20], dtype=torch.int64,
                                                      device=device)
    # TODO (Test): check other intermediate tensors and values, buffers, etc.

    iterations = 20
    for k in range(iterations):
        cluster_data = seqs[k % 4]
        context_data = context
        flock.forward_learn(cluster_data, context_data, input_rewards=None)

    assert same(expected_buffer_total_data_written, flock.buffer.total_data_written)

    # Run through the sequences yet again and no grab the outputs each time for flock 2
    for k in range(4):
        flock.forward_learn(seqs[k], context, input_rewards=None)
        # Check that the predicted outputs are what we expect
        assert same(expected_actions[k], flock.action_outputs, eps=1e-2)
        # Check that the output projection is what we expect
        assert same(expected_projections[k], flock.projection_outputs, eps=1e-3)


def test_forward_learn_context():
    """Tests if context helps to disambiguate situations."""
    device = 'cuda'
    float_dtype = get_float(device)
    batch_size = 5
    flock = create_tp_flock(flock_size=2, seq_length=2, buffer_size=1000, batch_size=batch_size,
                            max_encountered_seq=800, n_frequent_seq=50,
                            incoming_context_size=2, learning_period=4, context_prior=SMALL_CONSTANT,
                            forgetting_limit=1)

    # experts:
    #   ex1 does not receive the context during learning, so can't learn to predict correctly
    #   ex2 receives a cycling sequence and useful context - this one should learn appropriately

    seqs = torch.tensor([[[1, 0, 0], [1, 0, 0]],
                         [[0, 1, 0], [0, 1, 0]],
                         [[0, 0, 1], [0, 0, 1]],
                         [[0, 1, 0], [0, 1, 0]]], dtype=float_dtype, device=device)

    contexts = torch.tensor([[[[[1, 0], [0, 0], [0, 0]]], [[[1, 0], [0, 0], [0, 0]]]],
                             [[[[1, 0], [0, 0], [0, 0]]], [[[1, 0], [0, 0], [0, 0]]]],
                             [[[[1, 0], [0, 0], [0, 0]]], [[[1, 0], [0, 0], [0, 0]]]],
                             [[[[1, 0], [0, 0], [0, 0]]], [[[0, 1], [0, 0], [0, 0]]]]], dtype=float_dtype, device=device)

    # In ex1, the context doesnt help, in ex2, the context does
    expected_actions = torch.tensor([[[0, 1, 0], [0, 1, 0]],
                                     [[0.5, 0, 0.5], [0, 0, 1]],
                                     [[0, 1, 0], [0, 1, 0]],
                                     [[0.5, 0, 0.5], [1, 0, 0]]], dtype=float_dtype, device=device)

    expected_projections = torch.tensor([[[0.6666, 0.3333, 0.], [0.6666, 0.3333, 0.]],
                                         [[0.1666, 0.6666, 0.1666], [0, 0.6666, 0.3333]],
                                         [[0., 0.3333, 0.6666], [0., 0.3333, 0.6666]],
                                         [[0.1666, 0.6666, 0.1666], [0.3333, 0.6666, 0.]]], dtype=float_dtype,
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
        flock.forward_learn(seqs[k], contexts[k], input_rewards=None)
        # Check that the predicted outputs are what we expect
        assert same(expected_actions[k], flock.action_outputs, eps=1e-2)
        # Check that the output projection is what we expect
        assert same(expected_projections[k], flock.projection_outputs, eps=3e-2)

@pytest.mark.skip("Test obseleted")
def test_forward_learn_reward():
    """Tests if the agent tries to reach the reward.

    There are two sequences: 0, 1, 2 and 0, 3, 4.
    There can be a reward associated with cluster 3 or punishment associated with cluster 2.
    """
    device = 'cuda'
    float_dtype = get_float(device)
    data_length = 6
    batch_size = data_length + 2
    flock_size = 3
    n_cluster_centers = 5
    flock = create_tp_flock(flock_size=flock_size, seq_length=3, seq_lookahead=2, buffer_size=100,
                            batch_size=batch_size, max_encountered_seq=20, n_frequent_seq=10,
                            n_cluster_centers=n_cluster_centers, incoming_context_size=1,
                            learning_period=data_length, context_prior=0.001, forgetting_limit=1,
                            follow_goals=True)

    seqs = torch.tensor([[[1, 0, 0, 0, 0]],
                         [[0, 1, 0, 0, 0]],
                         [[0, 0, 1, 0, 0]],
                         [[1, 0, 0, 0, 0]],
                         [[0, 0, 0, 1, 0]],
                         [[0, 0, 0, 0, 1]]],
                        dtype=float_dtype, device=device).expand(data_length, flock_size, n_cluster_centers)

    # context does not help in this case
    contexts = torch.tensor([[[[1], [0.5]]]],
                            dtype=float_dtype, device=device).expand(data_length, flock_size, NUMBER_OF_CONTEXT_TYPES,
                                                                     1)

    # experts:
    #   ex1 does not receive the reward during learning, so can't learn to reach it
    #   ex2 receives the positive reward signal - this one should learn to approach it
    #   ex3 receives the negative reward signal (punishment) - this one should learn to avoid it
    rewards = torch.tensor([[[0, 0], [0, 0], [0, 0]],
                            [[0, 0], [0, 0], [0, 0]],
                            [[0, 0], [0, 0], [0, 1]],
                            [[0, 0], [0, 0], [0, 0]],
                            [[0, 0], [0, 0], [0, 0]],
                            [[0, 0], [1, 0], [0, 0]]])

    expected_actions = torch.tensor([[[0, 0.5, 0, 0.5, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]],
                                     [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
                                     [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
                                     [[0, 0.5, 0, 0.5, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]],
                                     [[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]],
                                     [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]], dtype=float_dtype,
                                    device=device)

    # need to learn at least 2 times because contexts are not extracted when sequence is new. And they are regarded
    #  as uniform context so we need even more to get to a sharp probability
    iterations = data_length * 4
    for k in range(iterations):
        cluster_data = seqs[k % data_length]
        context_data = contexts[k % data_length]
        rewards_data = rewards[k % data_length]
        flock.forward_learn(cluster_data, context_data, rewards_data)

    # Run through the sequences yet again and no grab the outputs each time for flock 2
    for k in range(data_length):
        flock.forward_learn(seqs[k], contexts[k], input_rewards=rewards[k])
        # Check that the predicted outputs are what we expect
        assert same(expected_actions[k], flock.action_rewards, eps=2e-2)


@pytest.mark.skip(reason="Ignoring non-default stream tests for now.")
def test_forward_learn_non_default_stream():
    device = 'cuda'
    float_dtype = get_float(device)
    flock = create_tp_flock(flock_size=2, seq_length=3, buffer_size=1000, batch_size=20, max_encountered_seq=800)

    # experts:
    #   ex1 receives the same input all the time - doesn't learn
    #   ex2 receives a cycling sequence - this one should learn appropriately

    seqs = torch.tensor([[[1, 0, 0], [1, 0, 0]],
                         [[1, 0, 0], [0, 1, 0]],
                         [[1, 0, 0], [0, 0, 1]],
                         [[1, 0, 0], [0, 1, 0]]], dtype=float_dtype, device=device)

    # If the forward pass doesnt run, the outputs should always be zero for ex1
    expected_outputs = torch.tensor([[[[0.3333, 0.3333, 0.3333]], [[0, 1, 0]]],
                                     [[[0.3333, 0.3333, 0.3333]], [[0, 0, 1]]],
                                     [[[0.3333, 0.3333, 0.3333]], [[0, 1, 0]]],
                                     [[[0.3333, 0.3333, 0.3333]], [[1, 0, 0]]]], dtype=float_dtype, device=device)

    expected_projections = torch.tensor([[[0.3333, 0.3333, 0.3333], [0.5, 0.5, 0]],
                                         [[0.3333, 0.3333, 0.3333], [0.25, 0.5, 0.25]],
                                         [[0.3333, 0.3333, 0.3333], [0, 0.5, 0.5]],
                                         [[0.3333, 0.3333, 0.3333], [0.25, 0.5, 0.25]]], dtype=float_dtype,
                                        device=device)

    with torch.cuda.stream(torch.cuda.Stream()):
        iterations = 20
        for k in range(iterations):
            data = seqs[k % 4]
            flock.forward_learn(data)

        # Run through the seqs yet again and no grab the outputs each time for flock 2
        for k in range(4):
            flock.forward_learn(seqs[k])
            # Check that the predicted outputs are what we expect
            assert same(expected_outputs[k], flock.action_rewards, eps=1e-4)
            # Check that the output projection is what we expect
            assert same(expected_projections[k], flock.projection_outputs, eps=1e-4)


@pytest.mark.parametrize('enable_learning', [True, False])
def test_forward_learn_enable_learning(enable_learning):
    device = 'cuda'
    float_dtype = get_float(device)
    flock = create_tp_flock(flock_size=1, seq_length=3, buffer_size=1000, batch_size=4, max_encountered_seq=800,
                            incoming_context_size=1, exploration_probability=0, enable_learning=enable_learning)

    seqs = torch.tensor([[[1, 0, 0]],
                         [[0, 1, 0]],
                         [[0, 0, 1]],
                         [[0, 1, 0]]], dtype=float_dtype, device=device)

    initial_seq_occurrences = flock.all_encountered_seq_occurrences.clone()

    iterations = 4
    for k in range(iterations):
        cluster_data = seqs[k % 4]
        flock.forward_learn(cluster_data, input_context=None, input_rewards=None)

    # should be different if enable_learning == True
    assert (not same(initial_seq_occurrences, flock.all_encountered_seq_occurrences)) == enable_learning


# TODO: merge this test with the SP one - actually, the whole determination process could be probablz moved to the
# TODO: base class
@pytest.mark.parametrize("mask, total_data_written, data_since_last_sample, expected_learn_tensor",
                         [(torch.ones(7, dtype=torch.uint8), 0, 0, torch.zeros(7, dtype=torch.uint8)),
                          (torch.ones(7, dtype=torch.uint8), 5, 5, torch.zeros(7, dtype=torch.uint8)),
                          (torch.ones(7, dtype=torch.uint8), 11, 1, torch.zeros(7, dtype=torch.uint8)),
                          (torch.ones(7, dtype=torch.uint8), 15, 5, torch.ones(7, dtype=torch.uint8)),
                          (torch.zeros(7, dtype=torch.uint8), 15, 5, torch.zeros(7, dtype=torch.uint8)),
                          (torch.tensor([0, 1, 1, 0, 0, 0, 1], dtype=torch.uint8), 15, 5,
                           torch.tensor([0, 1, 1, 0, 0, 0, 1], dtype=torch.uint8)),
                          (torch.tensor([0, 1, 1, 1, 0, 1, 1], dtype=torch.uint8),
                           torch.tensor([11, 9, 11, 11, 15, 8, 100], dtype=torch.uint8),
                           torch.tensor([5, 6, 4, 8, 5, 1, 5], dtype=torch.uint8),
                           torch.tensor([0, 0, 0, 1, 0, 0, 1], dtype=torch.uint8))])
def test_determine_learning(mask, total_data_written, data_since_last_sample, expected_learn_tensor):
    """See _test_determine_learning in tests.core.models.spatial_pooler.test_sp_processes."""
    params = ExpertParams()
    params.flock_size = 7
    params.n_cluster_centers = 5

    t_pooler = params.temporal
    t_pooler.batch_size = 10
    t_pooler.learning_period = 5

    flock = TPFlock(params, creator=AllocatingCreator('cpu'))
    copy_or_fill_tensor(flock.buffer.total_data_written, total_data_written)
    copy_or_fill_tensor(flock.buffer.data_since_last_sample, data_since_last_sample)
    learn_tensor = flock._determine_learning(mask)
    assert same(expected_learn_tensor, learn_tensor)


@pytest.mark.slow
def test_trained_untrained():
    flock_size = 2
    n_cluster_centers = 3
    batch_size = 2
    buffer_size = 5
    seq_length = 2
    device = 'cuda'
    float_dtype = get_float(device)

    trained_factory = TrainedForwardProcessFactory()
    untrained_factory = UntrainedForwardProcessFactory()

    # sequence of length 4 of input clusters (3) for each expert in the flock (2)
    input_clusters_seq = torch.tensor([
        [[1, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [0, 0, 1]]
    ], dtype=float_dtype, device=device)

    # The first expert:
    # Step 1: First input, run untrained process.
    # Step 2: Second input, run untrained process, learns for the first time.
    # Step 3: It already learned, and there's a new input, so the trained process should run.
    # Step 4: Run trained process again.

    # The second expert:
    # Step 1: First input, run untrained process.
    # Step 2: Same input, nothing runs.
    # Step 3: Second input, run untrained process, learns for the first time.
    # Step 4: It already learned, and there's a new input, so the trained process should run.

    # indices of trained experts that are ran in a given step
    expected_indices_trained_seq = [
        torch.tensor([[]], dtype=torch.long, device=device).permute(1, 0),
        # the expected shape of this is [0,1], not [1,0]
        torch.tensor([[]], dtype=torch.long, device=device).permute(1, 0),
        torch.tensor([[0]], dtype=torch.long, device=device),
        torch.tensor([[0], [1]], dtype=torch.long, device=device),
    ]

    # indices of untrained experts that are ran in a givne step
    expected_indices_untrained_seq = [
        torch.tensor([[0], [1]], dtype=torch.long, device=device),
        torch.tensor([[0]], dtype=torch.long, device=device),
        torch.tensor([[1]], dtype=torch.long, device=device),
        torch.tensor([[]], dtype=torch.long, device=device).permute(1, 0),
    ]

    def _last_called_args(mock):
        # this extracts parameters that are passed to the create method
        return [c for c in mock.mock_calls if c[0] == ''][-1][1]

    with patch.object(TrainedForwardProcessFactory,
                      'create',
                      wraps=trained_factory.create) as trained_mock:
        with patch.object(UntrainedForwardProcessFactory,
                          'create',
                          wraps=untrained_factory.create) as untrained_mock:
            pooler = create_tp_flock(flock_size=flock_size,
                                     n_cluster_centers=n_cluster_centers,
                                     batch_size=batch_size,
                                     buffer_size=buffer_size,
                                     seq_length=seq_length,
                                     trained_forward_factory=trained_factory,
                                     untrained_forward_factory=untrained_factory,
                                     device=device)

            trained_call_indices_seq = []
            untrained_call_indices_seq = []
            # for each sequence of input clusters
            for input_clusters in input_clusters_seq:
                pooler.forward_learn(input_clusters)

                # read the parameters passed to the create method
                last_trained_create_call = _last_called_args(trained_mock)
                last_untrained_create_call = _last_called_args(untrained_mock)

                # store just the indices with which the process was created.
                trained_call_indices_seq.append(last_trained_create_call[4])
                untrained_call_indices_seq.append(last_untrained_create_call[4])

    # test that the forward_learn process correctly determined the trained and untrained indices
    assert same(trained_call_indices_seq[0], expected_indices_trained_seq[0])
    assert same(untrained_call_indices_seq[0], expected_indices_untrained_seq[0])
    assert same(trained_call_indices_seq[1], expected_indices_trained_seq[1])
    assert same(untrained_call_indices_seq[1], expected_indices_untrained_seq[1])
    assert same(trained_call_indices_seq[2], expected_indices_trained_seq[2])
    assert same(untrained_call_indices_seq[2], expected_indices_untrained_seq[2])
    assert same(trained_call_indices_seq[3], expected_indices_trained_seq[3])
    assert same(untrained_call_indices_seq[3], expected_indices_untrained_seq[3])


class TestTPFlock:
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('data, seqs, likelihoods, n_top_sequences, expected_output', [
        (
                [[0, 0, 1, 0]],
                [[[0, 2, 0], [0, 2, 1]]],
                [[1.0, 0.5]],
                1,
                [[0.666, 0.0, 0.333, 0.0]]
        ),
        (
                [[0, 0, 1, 0]],
                [[[0, 2, 0], [0, 2, 1]]],
                [[0.5, 1.0]],
                1,
                [[0.333, 0.333, 0.333, 0.0]]
        ),
        (
                [[0, 0, 1, 0]],
                [[[0, 2, 0], [0, 2, 1], [0, 1, 3]]],
                [[0.5, 1.0, 0.5]],
                2,
                [[0.5000, 0.1667, 0.3333, 0.0000]]
        )
    ])
    def test_inverse_projection(self, data, seqs, likelihoods, n_top_sequences, expected_output, device):
        float_type = get_float(device)
        t_data = torch.tensor(data, dtype=float_type, device=device)
        t_seqs = torch.tensor(seqs, dtype=torch.int64, device=device)
        t_likelihoods = torch.tensor(likelihoods, dtype=float_type, device=device)
        t_expected_output = torch.tensor(expected_output, dtype=float_type, device=device)
        flock_size, n_frequent_seq, seq_length = t_seqs.shape
        n_cluster_centers = t_data.shape[-1]

        flock = create_tp_flock(flock_size=flock_size, seq_length=seq_length, seq_lookahead=1,
                                n_frequent_seq=n_frequent_seq, n_cluster_centers=n_cluster_centers, device=device)

        flock.frequent_seqs = t_seqs
        flock.frequent_seq_likelihoods_priors_clusters_context = t_likelihoods
        output = flock.inverse_projection(t_data, n_top_sequences=n_top_sequences)
        assert same(t_expected_output, output, eps=1e-3)
