import torch
from torchsim.core import get_float, SMALL_CONSTANT
from torchsim.core.models.expert_params import NUMBER_OF_CONTEXT_TYPES, EXPLORATION_REWARD, REWARD_DISCOUNT_FACTOR
from torchsim.core.models.temporal_pooler.buffer import TPFlockBuffer
from torchsim.core.models.temporal_pooler.kernels import tp_process_kernels
from torchsim.core.models.temporal_pooler.process import TPProcess
from torchsim.core.models.temporal_pooler.tp_output_projection import TPOutputProjection
from torchsim.core.utils.tensor_utils import normalize_probs, normalize_probs_, multi_unsqueeze, move_probs_towards_50, \
    weighted_sum_, kl_divergence, id_to_one_hot, safe_id_to_one_hot


class TPFlockForwardAndBackward(TPProcess):
    """The forward pass of temporal pooler."""
    _float_dtype: torch.dtype
    _output_projection: TPOutputProjection

    _buffer: TPFlockBuffer

    def __init__(self,
                 indices: torch.Tensor,
                 do_subflocking: bool,
                 buffer: TPFlockBuffer,
                 cluster_data: torch.Tensor,
                 context_data: torch.Tensor,
                 reward_data: torch.Tensor,
                 frequent_seqs: torch.Tensor,
                 frequent_seq_occurrences: torch.Tensor,
                 frequent_seq_likelihoods_priors_clusters_context: torch.Tensor,
                 frequent_context_likelihoods: torch.Tensor,
                 frequent_rewards_punishments: torch.Tensor,
                 frequent_exploration_attempts: torch.Tensor,
                 frequent_exploration_results: torch.Tensor,
                 projection_outputs: torch.Tensor,
                 action_outputs: torch.Tensor,
                 action_rewards: torch.Tensor,
                 action_punishments: torch.Tensor,
                 passive_predicted_clusters_outputs: torch.Tensor,
                 execution_counter: torch.Tensor,
                 seq_likelihoods_by_context: torch.Tensor,  # Both input and output tensor
                 best_matching_context: torch.Tensor,
                 n_frequent_seqs: int,
                 n_cluster_centers: int,
                 seq_length: int,
                 seq_lookahead: int,
                 context_size: int,
                 n_providers: int,
                 exploration_probability: float,
                 own_rewards_weight: float,
                 cluster_exploration_prob: float,
                 device: str,
                 follow_goals: bool,
                 convolutional: bool = False,
                 produce_actions: bool = False,
                 compute_backward_pass: bool = True,
                 compute_best_matching_context: bool = True):
        super().__init__(indices, do_subflocking)
        self._float_dtype = get_float(device)
        self.device = device

        self.n_frequent_seqs = n_frequent_seqs
        self.n_cluster_centers = n_cluster_centers

        self.seq_length = seq_length
        self.seq_lookahead = seq_lookahead
        self.seq_lookbehind = self.seq_length - self.seq_lookahead

        self.context_size = context_size
        self.n_providers = n_providers

        self.exploration_probability = exploration_probability
        self.follow_goals = follow_goals
        self.own_rewards_weight = own_rewards_weight
        self.cluster_exploration_prob = cluster_exploration_prob

        # Does this flock produce actions?
        self.produce_actions = produce_actions

        self.compute_backward_pass = compute_backward_pass
        self.compute_best_matching_context = compute_best_matching_context

        self.cluster_history = torch.zeros(self._flock_size, self.seq_lookbehind, self.n_cluster_centers,
                                           dtype=self._float_dtype, device=device)

        self.context_history = torch.zeros(self._flock_size, self.seq_lookbehind, self.n_providers,
                                           self.context_size,
                                           dtype=self._float_dtype, device=device)

        # P(S | C) - sequence likelihoods based just on current cluster history
        self.seq_likelihoods_clusters = torch.zeros(self._flock_size, self.n_frequent_seqs,
                                                    dtype=self._float_dtype, device=device)

        # P(S | Prior, C) - sequence likelihoods based just on prior probabilities and current cluster history
        self.seq_likelihoods_priors_clusters = torch.zeros(self._flock_size, self.n_frequent_seqs,
                                                           dtype=self._float_dtype, device=device)

        # P(S | C, Oi) - sequence likelihoods based on current cluster history and each
        # provider independently
        self.seq_likelihoods_for_each_provider = torch.zeros(self._flock_size, self.n_frequent_seqs,
                                                             self.seq_lookbehind, self.n_providers,
                                                             dtype=self._float_dtype, device=device)

        self.predicted_clusters_by_context = torch.zeros(self._flock_size, self.n_providers, self.context_size,
                                                         self.n_cluster_centers,
                                                         dtype=self._float_dtype, device=device)

        # P(S | C, O*) - sequence likelihoods based on, current cluster history and the most informative context
        self.seq_probs_clusters_context = torch.zeros(self._flock_size, self.n_frequent_seqs,
                                                      dtype=self._float_dtype, device=device)

        # region Backward

        # P(S | Prior, C, O*, how much I want to explore it) - sequence likelihoods based on
        # _seq_likelihoods_priors_clusters_context and how much the expert want them to be explored (
        self.seq_likelihoods_exploration = torch.zeros(self._flock_size, self.n_frequent_seqs,
                                                       dtype=self._float_dtype, device=device)

        self.seq_likelihoods_active = torch.zeros((self._flock_size, self.n_frequent_seqs),
                                                  dtype=self._float_dtype, device=device)

        self.active_predicted_clusters = torch.zeros((self._flock_size, self.seq_length, self.n_cluster_centers),
                                                     dtype=self._float_dtype, device=device)
        # indicator which experts are exploring this turn (1) vs acting according to the goalDirected behavior (0)
        self.exploring = torch.zeros((self._flock_size, 1), dtype=self._float_dtype, device=device)
        # endregion

        self.exploration_random_numbers = torch.zeros((self._flock_size, 1), dtype=self._float_dtype, device=device)

        self.context_informativeness = torch.zeros(self._flock_size, self.seq_lookbehind, self.n_providers,
                                                   dtype=self._float_dtype, device=device)

        self.frequent_seqs_scaled = torch.zeros((self._flock_size, self.n_frequent_seqs,
                                                 self.n_cluster_centers), dtype=self._float_dtype, device=device)

        self.seq_rewards_goal_directed = torch.zeros(
            (self._flock_size, self.n_frequent_seqs, NUMBER_OF_CONTEXT_TYPES - 1),
            dtype=self._float_dtype, device=device)

        self._influence_model = torch.zeros(
            (self._flock_size, self.n_frequent_seqs, self.seq_length - 1, self.n_cluster_centers),
            dtype=self._float_dtype, device=device)

        self._buffer = self._get_buffer(buffer)
        self._cluster_data = self._read(cluster_data)
        self._context_data = self._read(context_data)
        self._reward_data = self._read(reward_data)
        if convolutional:
            self._frequent_seqs = self._read_expanded(frequent_seqs).contiguous()
            self._frequent_seq_occurrences = self._read_expanded(frequent_seq_occurrences)
            self._frequent_context_likelihoods = self._read_expanded(frequent_context_likelihoods)
            self._frequent_exploration_attempts = self._read_expanded(frequent_exploration_attempts)
            self._frequent_exploration_results = self._read_expanded(frequent_exploration_results)
            self._frequent_rewards_punishments = self._read_expanded(frequent_rewards_punishments)
        else:
            self._frequent_seqs = self._read(frequent_seqs)
            self._frequent_seq_occurrences = self._read(frequent_seq_occurrences)
            self._frequent_context_likelihoods = self._read(frequent_context_likelihoods)
            self._frequent_exploration_attempts = self._read(frequent_exploration_attempts)
            self._frequent_exploration_results = self._read(frequent_exploration_results)
            self._frequent_rewards_punishments = self._read(frequent_rewards_punishments)
        self._passive_predicted_clusters = self._read_write(passive_predicted_clusters_outputs)
        self._projection_outputs = self._read_write(projection_outputs)
        self._action_outputs = self._read_write(action_outputs)
        self._action_rewards = self._read_write(action_rewards)
        self._action_punishments = self._read_write(action_punishments)
        self._execution_counter = self._read_write(execution_counter)
        self._seq_likelihoods_by_context = self._read_write(seq_likelihoods_by_context)
        self._best_matching_context = self._read_write(best_matching_context)

        # P(S | Prior, C, O*) - sequence likelihoods based on prior probabilities, current cluster history and the most
        #  informative context
        self.seq_likelihoods_priors_clusters_context = self._read_write(
            frequent_seq_likelihoods_priors_clusters_context)

        self._check_dims(self._cluster_data, self._context_data, self._frequent_seqs, self._frequent_seq_occurrences,
                         self._frequent_context_likelihoods, self._passive_predicted_clusters, self._projection_outputs,
                         self._action_outputs)

        self._output_projection = TPOutputProjection(self._flock_size, self.n_frequent_seqs, self.n_cluster_centers,
                                                     self.seq_length, self.seq_lookahead, self.device)

    def _check_dims(self, cluster_data: torch.Tensor,
                    context_data: torch.Tensor,
                    frequent_seqs: torch.Tensor,
                    frequent_seq_occurrences: torch.Tensor,
                    frequent_context_likelihoods: torch.Tensor,
                    passive_predicted_clusters: torch.Tensor,
                    projection_outputs: torch.Tensor,
                    action_outputs: torch.Tensor):
        assert cluster_data.size() == (self._flock_size, self.n_cluster_centers)
        assert context_data.size() == (self._flock_size, self.n_providers, NUMBER_OF_CONTEXT_TYPES, self.context_size)
        assert frequent_seqs.size() == (self._flock_size, self.n_frequent_seqs, self.seq_length)
        assert frequent_seq_occurrences.size() == (self._flock_size, self.n_frequent_seqs)
        assert frequent_context_likelihoods.size() == (
            self._flock_size, self.n_frequent_seqs, self.seq_length, self.n_providers, self.context_size)
        assert projection_outputs.size() == (self._flock_size, self.n_cluster_centers)
        assert action_outputs.size() == (self._flock_size, self.n_cluster_centers)
        assert passive_predicted_clusters.size() == (self._flock_size, self.seq_length,
                                                     self.n_cluster_centers)

    def run(self):
        """Runs the forward and backward passes.

        Calculates the sequence probabilities, predicted clusters and output projection.
        Stores inputs and results in the buffer.
        """
        with self._buffer.next_step():
            seq_likelihoods_passive = self._forward(self._buffer,
                                                    self._cluster_data,
                                                    self._context_data,
                                                    self._reward_data,
                                                    self._projection_outputs,
                                                    self._passive_predicted_clusters,
                                                    self.cluster_history,
                                                    self.context_history,
                                                    self.seq_likelihoods_clusters,
                                                    self.seq_likelihoods_priors_clusters,
                                                    self.seq_likelihoods_for_each_provider,
                                                    self._seq_likelihoods_by_context,
                                                    self.predicted_clusters_by_context,
                                                    self.seq_likelihoods_priors_clusters_context,
                                                    self._best_matching_context,
                                                    self.seq_probs_clusters_context,
                                                    self._frequent_seqs,
                                                    self._frequent_seq_occurrences,
                                                    self._frequent_context_likelihoods,
                                                    self.context_informativeness)

            if self.compute_backward_pass:

                self._backward(seq_likelihoods_passive,
                               self._context_data,
                               self._frequent_exploration_attempts,
                               self._frequent_exploration_results,
                               self.exploring,
                               self.exploration_random_numbers,
                               self.seq_likelihoods_exploration,
                               self.seq_likelihoods_active,
                               self.seq_likelihoods_priors_clusters_context,
                               self.seq_rewards_goal_directed,
                               self._frequent_context_likelihoods,
                               self._frequent_rewards_punishments,
                               self._frequent_seqs,
                               self._frequent_seq_occurrences,
                               self.active_predicted_clusters,
                               self._action_outputs,
                               self._action_rewards,
                               self._action_punishments,
                               self._influence_model,
                               self._buffer)
            else:
                # just fill some dummy actions and 0 exploring
                self._action_outputs.fill_(0)
                self._buffer.actions.store(self._action_outputs)

                self.exploring.fill_(0)
                self._buffer.exploring.store(self.exploring)

                self._action_rewards.fill_(0)
                self._action_punishments.fill_(0)

            self._execution_counter += 1

    def _compute_predicted_clusters(self,
                                    frequent_seqs: torch.Tensor,
                                    seq_likelihoods: torch.Tensor,
                                    predicted_clusters: torch.Tensor,
                                    cluster_rewards: torch.Tensor = None):
        """Calculate the past, current and predicted cluster probabilities based on sequence likelihoods."""
        # Convert each cluster center id to a set of one-hot vectors corresponding to the cluster in
        # the cluster center space.
        # [flock_size, n_frequent_seqs, seq_length, n_cluster_centers]
        frequent_seqs_unrolled = safe_id_to_one_hot(frequent_seqs, self.n_cluster_centers, self._float_dtype)

        # Expand sequence likelihoods so we can multiply the unrolled freq_seq with it
        seq_likelihoods_expanded = seq_likelihoods.view(self._flock_size, self.n_frequent_seqs, 1, 1)
        seq_likelihoods_expanded = seq_likelihoods_expanded.expand(self._flock_size, self.n_frequent_seqs,
                                                                   self.seq_length, 1)

        # Obtain likelihoods of clusters by multiplying by _seq_likelihoods and summing
        cluster_likelihoods = (frequent_seqs_unrolled * seq_likelihoods_expanded).sum(1)
        if cluster_rewards is not None:
            cluster_rewards.copy_(cluster_likelihoods[:, self.seq_lookbehind, :])

        # Normalise.
        normalize_probs_(cluster_likelihoods, 2)

        # Now copy the normalised likelihoods into the predicted_clusters
        predicted_clusters.copy_(cluster_likelihoods)

    # region Forward

    def _forward(self,
                 buffer: TPFlockBuffer,
                 cluster_data: torch.Tensor,
                 context_data: torch.Tensor,
                 reward_data: torch.Tensor,
                 projection_outputs: torch.Tensor,
                 passive_predicted_clusters_outputs: torch.Tensor,
                 cluster_history: torch.Tensor,
                 context_history: torch.Tensor,
                 seq_likelihoods_clusters: torch.Tensor,
                 seq_likelihoods_priors_clusters: torch.Tensor,
                 seq_likelihoods_for_each_context: torch.Tensor,
                 seq_likelihoods_by_context: torch.Tensor,
                 predicted_clusters_by_context: torch.Tensor,
                 seq_likelihoods_priors_clusters_context: torch.Tensor,
                 best_matching_context: torch.Tensor,
                 seq_probs_clusters_context: torch.Tensor,
                 frequent_seqs: torch.Tensor,
                 frequent_seq_occurrences: torch.Tensor,
                 frequent_context_likelihoods: torch.Tensor,
                 context_informativeness: torch.Tensor) -> torch.Tensor:
        # Add small constant to all inputs which mean probabilities
        buffer.clusters.store(normalize_probs(cluster_data, dim=1, add_constant=True))
        # Don't store the rewards and punishments part from the parent... It's not needed.
        buffer.contexts.store(move_probs_towards_50(context_data[:, :, 0, :]))
        # Do store rewards and punishments that _you_ got this timestep
        buffer.rewards_punishments.store(reward_data)

        if self.compute_best_matching_context:
            # Note seq_likelihoods_by_context if from the last step (not computed for the actual one)
            self._compute_predicted_clusters_for_seq_likelihoods_by_context(frequent_seqs,
                                                                            seq_likelihoods_by_context,
                                                                            predicted_clusters_by_context)

            self._compute_best_matching_context(cluster_data, predicted_clusters_by_context, best_matching_context)

        self._compute_seq_likelihoods(buffer,
                                      cluster_history,
                                      context_history,
                                      seq_likelihoods_clusters,
                                      seq_likelihoods_priors_clusters,
                                      seq_likelihoods_for_each_context,
                                      seq_likelihoods_by_context,
                                      seq_likelihoods_priors_clusters_context,
                                      seq_probs_clusters_context,
                                      frequent_seqs,
                                      frequent_seq_occurrences,
                                      frequent_context_likelihoods,
                                      context_informativeness)

        # self._delete_improbable_seqs()

        # decide if we want to use context or without
        passive_seq_likelihoods = seq_likelihoods_priors_clusters_context

        buffer.seq_probs.store(normalize_probs(passive_seq_likelihoods, dim=1))

        self._apply_output_projection(frequent_seqs, passive_seq_likelihoods,
                                      projection_outputs)

        self._compute_predicted_clusters(frequent_seqs, passive_seq_likelihoods,
                                         passive_predicted_clusters_outputs)

        buffer.outputs.store(projection_outputs)

        return passive_seq_likelihoods

    def _compute_predicted_clusters_for_seq_likelihoods_by_context(self,
                                                                   frequent_seqs: torch.Tensor,
                                                                   seq_likelihoods_by_context: torch.Tensor,
                                                                   predicted_clusters_by_context: torch.Tensor
                                                                   ):
        """
        Compute predicted cluster likelihood for each context.

        Args:
            frequent_seqs: ['flock_size', 'tp_n_frequent_seqs', 'tp_seq_length'] -> 'cluster_id'
            seq_likelihoods_by_context: ['flock_size', 'tp_n_frequent_seqs', 'n_providers', 'context_size']
            predicted_clusters_by_context: ['flock_size', 'n_providers', 'context_size', 'n_cluster_centers']
        """
        predicted_cluster_output = torch.empty((self._flock_size, self.seq_length, self.n_cluster_centers),
                                               dtype=self._float_dtype, device=self.device)
        for context_id in range(self.context_size):
            for provider_id in range(self.n_providers):
                # [flock_size, n_frequent_seqs]
                context_likelihoods = seq_likelihoods_by_context[:, :, provider_id, context_id]
                if context_likelihoods.sum() == 0:
                    # Note: correct would be to treat each expert in the flock separately
                    # (i.e. determine all zeros per dim = 0). However, zeros are now just in the first step, this is OK.
                    context_likelihoods += SMALL_CONSTANT
                self._compute_predicted_clusters(frequent_seqs, context_likelihoods, predicted_cluster_output)
                # [flock_size, n_cluster_centers]
                predicted_cluster_output_next_symbol = predicted_cluster_output[:, self.seq_lookbehind, :]
                predicted_clusters_by_context[:, provider_id, context_id, :] = predicted_cluster_output_next_symbol

    def _compute_best_matching_context(self,
                                       cluster_data: torch.Tensor,
                                       predicted_clusters_by_context: torch.Tensor,
                                       best_matching_context: torch.Tensor):
        """
        Args:
            cluster_data: ['flock_size', 'n_cluster_centers']
            predicted_clusters_by_context: ['flock_size', 'n_providers', 'context_size', 'n_cluster_centers']
            best_matching_context: ['flock_size', 'n_providers', 'context_size']
        """
        input_data_expanded = multi_unsqueeze(cluster_data, [1, 2]).expand(predicted_clusters_by_context.shape)
        multiplied = input_data_expanded * predicted_clusters_by_context
        summed = multiplied.sum(dim=3)
        normalize_probs_(summed, dim=2, add_constant=True)
        best_matching_context.copy_(summed)

    def _compute_seq_likelihoods(self,
                                 buffer: TPFlockBuffer,
                                 cluster_history: torch.Tensor,
                                 context_history: torch.Tensor,
                                 seq_likelihoods_clusters: torch.Tensor,
                                 seq_likelihoods_priors_clusters: torch.Tensor,
                                 seq_likelihoods_for_each_context: torch.Tensor,
                                 seq_likelihoods_by_context: torch.Tensor,
                                 seq_likelihoods_priors_clusters_context: torch.Tensor,
                                 seq_probs_clusters_context: torch.Tensor,
                                 frequent_seqs: torch.Tensor,
                                 frequent_seq_occurrences: torch.Tensor,
                                 frequent_context_likelihoods: torch.Tensor,
                                 context_informativeness: torch.Tensor):
        """Calculates the sequence likelihoods."""

        self._compute_seq_likelihoods_priors_clusters(buffer,
                                                      cluster_history,
                                                      seq_likelihoods_clusters,
                                                      seq_likelihoods_priors_clusters,
                                                      frequent_seqs,
                                                      frequent_seq_occurrences)

        self._compute_seq_likelihoods_for_each_provider(buffer,
                                                        context_history,
                                                        seq_likelihoods_priors_clusters,
                                                        frequent_context_likelihoods,
                                                        seq_likelihoods_for_each_context,
                                                        seq_likelihoods_by_context)

        self._compute_provider_informativeness(seq_likelihoods_priors_clusters,
                                               seq_likelihoods_for_each_context,
                                               context_informativeness)

        # Uses the most informative context.
        self._compute_seq_likelihoods_priors_clusters_context(seq_likelihoods_priors_clusters_context,
                                                              seq_likelihoods_for_each_context,
                                                              context_informativeness)

        self._compute_seq_probs_without_priors(seq_likelihoods_priors_clusters_context,
                                               frequent_seq_occurrences,
                                               seq_probs_clusters_context)

    def _compute_seq_likelihoods_priors_clusters(self,
                                                 buffer: TPFlockBuffer,
                                                 cluster_history: torch.Tensor,
                                                 seq_likelihoods_clusters: torch.Tensor,
                                                 seq_likelihoods_priors_clusters: torch.Tensor,
                                                 frequent_seqs: torch.Tensor,
                                                 frequent_seq_occurrences: torch.Tensor):
        """Calculates sequence likelihoods based clusters only, and based on clusters with priors."""
        buffer.clusters.sample_forward_batch(self.seq_lookbehind, out=cluster_history)

        # compute the likelihoods based on clusters only
        tp_process_kernels.compute_seq_likelihoods_clusters_only(cluster_history,
                                                                 frequent_seqs,
                                                                 frequent_seq_occurrences,
                                                                 seq_likelihoods_clusters,
                                                                 self._flock_size,
                                                                 self.n_frequent_seqs,
                                                                 self.seq_lookbehind)

        # multiply by priors to get the likelihoods without context
        torch.mul(seq_likelihoods_clusters, frequent_seq_occurrences, out=seq_likelihoods_priors_clusters)

    def _compute_seq_likelihoods_for_each_provider(self,
                                                   buffer: TPFlockBuffer,
                                                   context_history: torch.Tensor,
                                                   seq_likelihoods_priors_clusters: torch.Tensor,
                                                   frequent_context_likelihoods: torch.Tensor,
                                                   seq_likelihoods_for_each_provider: torch.Tensor,
                                                   seq_likelihoods_by_context: torch.Tensor):
        """Calculates sequence likelihoods based on context history and cluster history.

        Calculates independently for each parent.

        Args:
            seq_likelihoods_for_each_provider - Output
        """

        buffer.contexts.sample_forward_batch(self.seq_lookbehind, out=context_history)

        # Expand the history over the lookbehind of frequent context probs so they can be multiplied.
        context_history_expanded = context_history.unsqueeze(dim=1).expand(self._flock_size,
                                                                           self.n_frequent_seqs,
                                                                           self.seq_lookbehind,
                                                                           self.n_providers,
                                                                           self.context_size)

        likelihoods = frequent_context_likelihoods[:, :, :self.seq_lookbehind, :, :] * context_history_expanded
        # get rid of nans for the contexts which are shorten than the max context size and sum over context size
        likelihoods.masked_fill_(torch.isnan(likelihoods), 0)
        likelihoods = likelihoods.sum(dim=4)

        # Multiply by the likelihoods taken from the history of clusters and priors to get the real distribution of
        # sequences
        likelihoods *= multi_unsqueeze(seq_likelihoods_priors_clusters, [2, 3]).expand(likelihoods.size())

        # Take the sequence lookbehind
        seq_likelihoods_for_each_provider.copy_(likelihoods)

        # compute
        likelihoods_by_context = frequent_context_likelihoods[:, :, self.seq_lookbehind, :, :].clone()
        likelihoods_by_context.masked_fill_(torch.isnan(likelihoods_by_context), 0)
        likelihoods_by_context *= multi_unsqueeze(seq_likelihoods_priors_clusters, [2, 3]).expand(likelihoods_by_context.size())

        # result = normalize_probs(likelihoods_by_context, dim=4, add_constant=False)
        seq_likelihoods_by_context.copy_(likelihoods_by_context)

    @staticmethod
    def _compute_provider_informativeness(seq_likelihoods_priors_clusters: torch.Tensor,
                                          seq_likelihoods_for_each_provider: torch.Tensor,
                                          provider_informativeness: torch.Tensor):
        """Calculates the KL divergence between the prob. dist. over seqs without context and with each context."""

        # At this point we expect that there are no exactly 0 (except for invalid frequent sequencies) or
        # exactly 1 values in the probabilities, so we do not add small constant here. If we did, it would cause
        # the invalid frequent sequences to have probabilities > 0.

        normalized_baseline = multi_unsqueeze(normalize_probs(seq_likelihoods_priors_clusters, dim=1,
                                                              add_constant=False), [2, 3]).expand(seq_likelihoods_for_each_provider.size())

        normalized_seq_likelihoods_for_each_context = normalize_probs(seq_likelihoods_for_each_provider, dim=1,
                                                                      add_constant=False)

        kl_divergence(normalized_baseline, normalized_seq_likelihoods_for_each_context,
                      output=provider_informativeness, dim=1)

    def _compute_seq_likelihoods_priors_clusters_context(self,
                                                         seq_likelihoods_priors_clusters_context: torch.Tensor,
                                                         seq_likelihoods_for_each_context: torch.Tensor,
                                                         context_informativeness: torch.Tensor):
        """Pick the most informative context for each expert and store it as seq_likelihoods_priors_clusters_context.
        Args:
            seq_likelihoods_priors_clusters_context: Output
        """

        # Find maximum over both last dimensions: lookbehind and context_size
        most_informative_context = torch.argmax(context_informativeness.view(self._flock_size, -1), dim=1)

        each_context_viewed = seq_likelihoods_for_each_context.view(
            self._flock_size, self.n_frequent_seqs, -1)

        indexes = most_informative_context.view(self._flock_size, 1, 1).expand(self._flock_size, self.n_frequent_seqs,
                                                                               1)

        torch.gather(input=each_context_viewed, dim=2, index=indexes,
                     out=seq_likelihoods_priors_clusters_context.unsqueeze(2))

    @staticmethod
    def _compute_seq_probs_without_priors(seq_likelihoods_priors_clusters_context: torch.Tensor,
                                          frequent_seq_occurrences: torch.Tensor,
                                          seq_probs_clusters_context: torch.Tensor):
        # Compute the sequence probabilities without the priors
        torch.div(input=seq_likelihoods_priors_clusters_context,
                  other=frequent_seq_occurrences,
                  out=seq_probs_clusters_context)

        # ... and get rid of any nans (invalid freq_seqs have 0 occurrences)
        seq_probs_clusters_context.masked_fill_(torch.isnan(seq_probs_clusters_context), 0)

        # normalize to get probabilities
        normalize_probs_(seq_probs_clusters_context, dim=1)

    def _apply_output_projection(self,
                                 frequent_seqs: torch.Tensor,
                                 # [flock_size, n_frequent_seqs, seq_length]
                                 seq_likelihoods: torch.Tensor,
                                 outputs: torch.Tensor):
        """Transfer the sequence likelihoods into the cluster space with more weight on clusters near in time."""
        self._output_projection.compute_output_projection(frequent_seqs, seq_likelihoods, outputs)

    # endregion

    # region Backward

    def _backward(self,
                  seq_likelihoods_passive: torch.Tensor,
                  context_data: torch.Tensor,
                  frequent_exploration_attempts: torch.Tensor,
                  frequent_exploration_results: torch.Tensor,
                  exploring: torch.Tensor,
                  exploration_random_numbers: torch.Tensor,
                  seq_likelihoods_exploration: torch.Tensor,
                  seq_likelihoods_active: torch.Tensor,
                  seq_likelihoods_priors_clusters_context: torch.Tensor,
                  seq_rewards_goal_directed: torch.Tensor,
                  frequent_context_likelihoods: torch.Tensor,
                  frequent_rewards_punishments: torch.Tensor,
                  frequent_seqs: torch.Tensor,
                  frequent_seq_occurrences: torch.Tensor,
                  active_predicted_clusters: torch.Tensor,
                  action_outputs: torch.Tensor,
                  action_rewards: torch.Tensor,
                  action_punishments: torch.Tensor,
                  influence_model: torch.Tensor,
                  buffer: TPFlockBuffer):

        # The influence model is the normalised probabilities of the exploration results
        influence_model[:, :, -self.seq_lookahead:] = normalize_probs(frequent_exploration_results, dim=3)

        # Normalise the sequence likelihoods to obtain probabilities
        seq_probs_priors_clusters_context = normalize_probs(seq_likelihoods_priors_clusters_context, dim=1)


        # NOTE: Punishements (and thus summations with rewards) are disabled
        # seq_likelihoods_goal_directed = seq_rewards_goal_directed.sum(dim=2)

        # decide which likelihoods to use passive vs goal_directed
        if self.follow_goals:
            self._compute_rewards(frequent_seqs,
                                  frequent_seq_occurrences,
                                  context_data,
                                  frequent_context_likelihoods,
                                  seq_probs_priors_clusters_context,
                                  frequent_rewards_punishments,
                                  influence_model,
                                  seq_rewards_goal_directed)

            # We only need the rewards here
            seq_likelihoods_active.copy_(seq_rewards_goal_directed[:, :, 0])
        else:
            seq_likelihoods_active.copy_(seq_likelihoods_passive)

        self._exploration_sequence(exploring, exploration_random_numbers, seq_likelihoods_active, buffer)

        self._compute_predicted_clusters(frequent_seqs,
                                         seq_likelihoods_active,
                                         active_predicted_clusters,
                                         action_rewards)

        # Store the actions that we care about
        # NOTE: Punishments to be considered in the future
        #
        # action_outputs.copy_(id_to_one_hot(torch.argmax(active_predicted_clusters[:, self.seq_lookbehind, :]), vector_len=self.n_cluster_centers))
        action_outputs.copy_(active_predicted_clusters[:, self.seq_lookbehind, :])

        # decide if exploring and if yes, overwrite the action_outputs
        self._exploration_cluster(exploration_random_numbers, action_outputs, action_rewards)

        # Store punishments to the punishments part too
        # NOTE: Punishments here are for sequences,  should be converted to clusters first
        # action_punishments.copy_(seq_rewards_goal_directed[:, :, 1])

        # store it to buffer
        buffer.actions.store(action_outputs)

    def _exploration_sequence(self, exploring: torch.Tensor, exploration_random_numbers: torch.Tensor,
                             sequence_likelihoods_active: torch.Tensor, buffer: TPFlockBuffer):
        """Decide if exploring. If yes, overwrite the sequence_likelihoods by a random sequence which will get
         converted to a corresponding cluster center later. Update the rewards with a corresponding exploration reward.

           Also stores the exploration flag into the buffer."""

        # Randomly decide which experts will be exploring
        exploration_random_numbers.uniform_(0, 1.0)
        exploring.copy_(exploration_random_numbers < self.exploration_probability)

        buffer.exploring.store(exploring)

        random_indices = torch.randint(low=0, high=self.n_frequent_seqs, size=(self._flock_size,), device=self.device)
        # one-hot representation of the sequences we want to explore right now
        exploration_sequences = id_to_one_hot(random_indices, self.n_frequent_seqs, dtype=self._float_dtype)

        exploration_sequences *= EXPLORATION_REWARD

        # overwrite the actions for experts which are exploring by exploration actions
        weighted_sum_(tensor_a=exploration_sequences, weight_a=exploring,
                      tensor_b=sequence_likelihoods_active, weight_b=1.0 - exploring,
                      output=sequence_likelihoods_active)

    def _exploration_cluster(self, exploration_random_numbers: torch.Tensor,
                             action_outputs: torch.Tensor, action_rewards: torch.Tensor):
        """For all those exploring, they have a self.exploration_probability chance of a totally random cluster being picked to explore instead
         of the sequence they were going to do in _exploration_sequence. Update rewards accordingly.

         """
        cluster_exploration = (exploration_random_numbers < self.cluster_exploration_prob * self.exploration_probability).float()

        random_indices = torch.randint(low=0, high=self.n_cluster_centers, size=(self._flock_size,), device=self.device)
        # one-hot representation of the actions (=clusters) we want to explore right now
        exploration_actions = id_to_one_hot(random_indices, self.n_cluster_centers, dtype=self._float_dtype)
        exploration_rewards = exploration_actions * EXPLORATION_REWARD

        # overwrite the actions for experts which are exploring by exploration actions
        weighted_sum_(tensor_a=exploration_actions, weight_a=cluster_exploration,
                      tensor_b=action_outputs, weight_b=1.0 - cluster_exploration,
                      output=action_outputs)

        # Do the same for the rewards
        weighted_sum_(tensor_a=exploration_rewards, weight_a=cluster_exploration,
                      tensor_b=action_rewards, weight_b=1.0 - cluster_exploration,
                      output=action_rewards)

    # endregion

    def _compute_rewards(self,
                         frequent_sequences: torch.Tensor,  # [flock_size, n_frequent_seqs, seq_len]
                         frequent_seq_occurrences: torch.Tensor,
                         context_data: torch.Tensor,  # [flock_size, n_providers, 3, context_size]
                         frequent_context_likelihoods: torch.Tensor,  # [flock_size, n_frequent_seqs, seq_length, n_providers, context_size)
                         seq_probs_priors_clusters_context: torch.Tensor,
                         frequent_rewards_punishments: torch.Tensor, # [flock_size, n_frequent_seqs, seq_length, 2]
                         influence_model: torch.Tensor,  # [flock_size, n_frequent_seqs, seq_lookahead, n_cluster_centers]
                         seq_rewards_goal_directed: torch.Tensor):  # [flock_size, n_frequent_seqs, 2]
        """Computes the expected rewards from each context provider for the next transition of all frequent sequences.

        We traverse the future clusters backwards using the current rewards context and remembered context probs.
        We calculate the expected rewards for reaching each cluster in each sequence using the current rewards context,
        then proceed to iteratively work backwards through the lookahead of the sequence updating the expected rewards
        by introducing the influence model and applying its effects to our model of the expected values.

        Once the expected rewards are calculated for all the lookahead transitions of each sequence for each provider,
        the rewards are discounted by the prior likelihoods and the expected rewards for the next step are saved."""
        influence_model[torch.isnan(influence_model)] = 0
        context_data[torch.isnan(context_data)] = 0

        # Unsqueeze and expand both the context data and the context on probs so that they can be multiplied
        context_data_rewards_punishments = multi_unsqueeze(context_data[:, :, 1:],
                                                           [1, 2]).expand(self._flock_size, self.n_frequent_seqs,
                                                                          self.seq_length, self.n_providers, 2,
                                                                          self.context_size)

        context_on_probs = frequent_context_likelihoods.unsqueeze(dim=4).expand(self._flock_size, self.n_frequent_seqs,
                                                                                self.seq_length, self.n_providers, 2,
                                                                                self.context_size)

        # Calculate the rewards/punishments for each cluster in each freq_seq lookahead assuming every
        # transition is perfect. AKA the undiscounted rewards from the parents.
        current_rewards = (context_data_rewards_punishments * context_on_probs).sum(dim=5)

        # Get the average freq_reward that this expert has seen by dividing the seen rewards by the occurrences
        frequent_rewards_punishments_scaled =\
            frequent_rewards_punishments / multi_unsqueeze(frequent_seq_occurrences.float() + SMALL_CONSTANT, [2, 3]).expand(frequent_rewards_punishments.size())

        # Expand the scaled rewards for addition to the current rewards
        # NOTE: This can be done in extract frequent sequence in the learning process instead of here
        frequent_rewards_punishments_scaled = frequent_rewards_punishments_scaled.unsqueeze(dim=3).expand(self._flock_size, self.n_frequent_seqs, self.seq_lookahead,
                                                                 self.n_providers, 2) * self.own_rewards_weight

        # Add in the scaled rewards to the lookahead part of the current rewards.
        current_rewards[:, :, self.seq_lookbehind:, :, :] += frequent_rewards_punishments_scaled

        # We iterate backwards through the possibilities of sequences to transform the undiscounted cluster rewards into
        # expected_values of following transitions

        # Allocate some temp storage for the kernel - This is the expected rewards/punishments for each destination
        # cluster when following the sequence from the current cluster, for all parents and all sequences
        cluster_rewards = torch.zeros((self._flock_size, self.n_frequent_seqs, self.n_providers, 2,
                                       self.n_cluster_centers), dtype=self._float_dtype, device=self.device)

        n_transitions = self.seq_length - 1
        # If this is the bottom level, we process all lookahead transitions, otherwise only down to
        # sequence_lookahead - 1. We do this so that higher levels of abstraction don't 'double discount' the expected
        # rewards of the lowest level for the next transition.
        if self.produce_actions:
            last_processed_transition = n_transitions - self.seq_lookahead
        else:
            last_processed_transition = n_transitions - (self.seq_lookahead - 1)

        # Now we iterate over the transitions of the sequence, starting from the last one (at n_transitions),
        # and moving towards the current point in the sequence (marked by last_processed_transition). The indices
        # for processing this range are a bit funky, as the index for the final transition is n_transitions-1, and the
        # true last_processed_transition index is last_processed_transition - 1.
        for transition in range(n_transitions - 1, last_processed_transition - 1, -1):
            cluster_rewards.fill_(0)

            tp_process_kernels.discount_rewards_iterative(frequent_sequences,
                                                          seq_probs_priors_clusters_context,
                                                          current_rewards,
                                                          influence_model,
                                                          cluster_rewards,
                                                          self._flock_size,
                                                          self.n_frequent_seqs,
                                                          self.n_cluster_centers,
                                                          transition,
                                                          self.n_providers)
            # Get the EV of trying to get to the next cluster of this sequence.
            transition_rewards = cluster_rewards.sum(dim=4) * REWARD_DISCOUNT_FACTOR

            # Update current rewards with the new EV of this transition (using the EV of rewards gained
            # when attempting this transition)
            current_rewards[:, :, transition + 1] = transition_rewards

            # Take max of transition EV vs origin cluster EV (for each provider) and update current rewards with it
            # Creates lower bound for EV
            max_ev = torch.max(current_rewards[:, :, transition], current_rewards[:, :, transition + 1])
            current_rewards[:, :, transition] = max_ev

        # Scale all rewards by their prior sequence probabilities
        discounted_rewards = current_rewards * multi_unsqueeze(seq_probs_priors_clusters_context,
                                                               [2, 3, 4]).expand(self._flock_size, self.n_frequent_seqs,
                                                                                 self.seq_length, self.n_providers, 2)

        # we care about the rewards and punishments for this transition, so get them
        next_potentials = discounted_rewards[:, :, self.seq_lookbehind]

        # Find max over all providers in each sequence in the influence_model
        next_potentials, _ = torch.max(next_potentials, dim=2)

        # Copy this into the return
        seq_rewards_goal_directed.copy_(next_potentials)
