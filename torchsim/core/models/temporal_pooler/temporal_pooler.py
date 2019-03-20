from abc import ABC, abstractmethod
from typing import Optional, Type

import torch

from torchsim.core import SMALL_CONSTANT
from torchsim.core.exceptions import IllegalArgumentException
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ExpertParams, NUMBER_OF_CONTEXT_TYPES, EXPLORATION_REWARD
from torchsim.core.models.temporal_pooler.forward_and_backward import TPFlockForwardAndBackward
from torchsim.core.models.temporal_pooler.learning import TPFlockLearning, ConvTPFlockLearning
from torchsim.core.models.temporal_pooler.tp_output_projection import TPOutputProjection
from torchsim.core.models.temporal_pooler.untrained_forward_and_backward import TPFlockUntrainedForwardAndBackward
from torchsim.core.utils.tensor_utils import detect_qualitative_difference, id_to_one_hot, normalize_probs_, \
    safe_id_to_one_hot, negate
from torchsim.gui.validators import validate_predicate
from .buffer import TPFlockBuffer
from ..flock import Flock


class ForwardProcessFactory(ABC):

    @abstractmethod
    def create(self,
               flock: 'TPFlock',
               data: torch.Tensor,
               context: torch.Tensor,
               rewards: torch.Tensor,
               indices: torch.Tensor,
               device: str):
        pass


class TrainedForwardProcessFactory(ForwardProcessFactory):

    _convolutional: bool

    def __init__(self, convolutional: bool = False):
        self._convolutional = convolutional

    def create(self,
               flock: 'TPFlock',
               data: torch.Tensor,
               context: torch.Tensor,
               rewards: torch.Tensor,
               indices: torch.Tensor,
               device: str) -> Optional[TPFlockForwardAndBackward]:
        # Guard to make sure indices has at least one element for forward pass
        if indices.size(0) == 0:
            # There is no forward pass happening, just return
            return None

        do_subflocking = indices.size(0) != flock.flock_size

        forward_process = TPFlockForwardAndBackward(indices,
                                                    do_subflocking,
                                                    flock.buffer,
                                                    data,
                                                    context,
                                                    rewards,
                                                    flock.frequent_seqs,
                                                    flock.frequent_seq_occurrences,
                                                    flock.frequent_seq_likelihoods_priors_clusters_context,
                                                    flock.frequent_context_likelihoods,
                                                    flock.frequent_rewards_punishments,
                                                    flock.frequent_exploration_attempts,
                                                    flock.frequent_exploration_results,
                                                    flock.projection_outputs,
                                                    flock.action_outputs,
                                                    flock.action_rewards,
                                                    flock.action_punishments,
                                                    flock.passive_predicted_clusters_outputs,
                                                    flock.execution_counter_forward,
                                                    flock.seq_likelihoods_by_context,
                                                    flock.best_matching_context,
                                                    flock.n_frequent_seqs,
                                                    flock.n_cluster_centers,
                                                    flock.seq_length,
                                                    flock.seq_lookahead,
                                                    flock.context_size,
                                                    flock.n_providers,
                                                    flock.exploration_probability,
                                                    flock.own_rewards_weight,
                                                    flock.cluster_exploration_prob,
                                                    device,
                                                    follow_goals=flock.follow_goals,
                                                    convolutional=self._convolutional,
                                                    produce_actions=flock.produce_actions,
                                                    compute_backward_pass=flock.compute_backward_pass,
                                                    compute_best_matching_context=flock.compute_best_matching_context
                                                    )

        return forward_process


class UntrainedForwardProcessFactory(ForwardProcessFactory):

    def create(self,
               flock: 'TPFlock',
               data: torch.Tensor,
               context: torch.Tensor,
               rewards: torch.Tensor,
               indices: torch.Tensor,
               device: str):
        # Guard to make sure indices has at least one element for forward pass
        if indices.size(0) == 0:
            # There is no forward pass happening, just return
            return None

        do_subflocking = indices.size(0) != flock.flock_size

        forward_process = TPFlockUntrainedForwardAndBackward(indices,
                                                             do_subflocking,
                                                             flock.buffer,
                                                             data,
                                                             context,
                                                             rewards,
                                                             flock.projection_outputs,
                                                             flock.action_rewards,
                                                             flock.n_frequent_seqs,
                                                             flock.n_cluster_centers,
                                                             device)

        return forward_process


class TPFlock(Flock):
    """A flock of temporal poolers.

    Discovers sequences in the incoming stream of spatial data.
    The standard use is together with a SPFlock of the same size.
    """
    trained_forward_process: TPFlockForwardAndBackward = None
    learn_process: TPFlockLearning = None
    _learning_process_class: Type[TPFlockLearning]

    def __init__(self,
                 params: ExpertParams,
                 creator: TensorCreator = torch,
                 trained_forward_factory=None,
                 untrained_forward_factory=None):
        super().__init__(params, creator.device)

        self.n_cluster_centers = params.n_cluster_centers
        self.flock_size = params.flock_size
        self.enable_learning = params.temporal.enable_learning
        self.produce_actions = params.produce_actions
        self.compute_backward_pass = params.temporal.compute_backward_pass
        self.compute_best_matching_context = params.temporal.compute_best_matching_context

        tp_params = params.temporal
        self.context_size = tp_params.incoming_context_size
        self.seq_length = tp_params.seq_length
        self.buffer_size = tp_params.buffer_size
        self.batch_size = tp_params.batch_size
        self.seq_lookahead = tp_params.seq_lookahead
        self.seq_lookbehind = tp_params.seq_lookbehind
        self.n_frequent_seqs = tp_params.n_frequent_seqs
        self.max_encountered_seqs = tp_params.max_encountered_seqs
        self.forgetting_limit = tp_params.forgetting_limit
        self.learning_period = tp_params.learning_period
        self._context_prior = tp_params.context_prior
        self.exploration_probability = tp_params.exploration_probability
        self.follow_goals = tp_params.follow_goals
        self.exploration_attempts_prior = tp_params.exploration_attempts_prior
        self.max_new_seqs = tp_params.max_new_seqs
        self.n_providers = tp_params.n_providers
        self.own_rewards_weight = tp_params.own_rewards_weight
        self.frustration_threshold = tp_params.frustration_threshold
        self.cluster_exploration_prob = tp_params.cluster_exploration_prob

        self.n_subbatches = tp_params.n_subbatches

        self._trained_forward_process_factory = trained_forward_factory or TrainedForwardProcessFactory()
        self._untrained_forward_process_factory = untrained_forward_factory or UntrainedForwardProcessFactory()

        self._learning_process_class = TPFlockLearning

        self.buffer = TPFlockBuffer(creator,
                                    self.flock_size,
                                    self.buffer_size,
                                    self.n_cluster_centers,
                                    self.n_frequent_seqs,
                                    self.context_size,
                                    self.n_providers)

        freq_all_enc_flock_size = self.get_frequent_all_enc_flock_size()

        self.all_encountered_seqs = creator.empty((freq_all_enc_flock_size, self.max_encountered_seqs, self.seq_length),
                                                  dtype=creator.int64, device=self._device)
        self.all_encountered_seq_occurrences = creator.empty((freq_all_enc_flock_size, self.max_encountered_seqs),
                                                             dtype=self._float_dtype, device=self._device)

        self.frequent_seqs = creator.empty((freq_all_enc_flock_size, self.n_frequent_seqs, self.seq_length),
                                           dtype=creator.int64, device=self._device)
        self.frequent_seq_occurrences = creator.empty((freq_all_enc_flock_size, self.n_frequent_seqs),
                                                      dtype=self._float_dtype, device=self._device)
        # P(S | Prior, C, O*) - sequence likelihoods based on prior probabilities, current cluster history and the most
        #  informative context
        self.frequent_seq_likelihoods_priors_clusters_context = creator.empty((self.flock_size, self.n_frequent_seqs),
                                                                              dtype=self._float_dtype,
                                                                              device=self._device)

        self.frequent_context_likelihoods = creator.empty(
            (freq_all_enc_flock_size, self.n_frequent_seqs, self.seq_length, self.n_providers, self.context_size),
            dtype=self._float_dtype, device=self._device)

        self.all_encountered_context_occurrences = creator.empty(
            (freq_all_enc_flock_size, self.max_encountered_seqs, self.seq_length, self.n_providers, self.context_size),
            dtype=self._float_dtype, device=self._device)

        self.frequent_exploration_attempts = creator.empty((freq_all_enc_flock_size, self.n_frequent_seqs, self.seq_lookahead),
                                                           dtype=self._float_dtype, device=self._device)

        self.frequent_exploration_results = creator.empty(
            (freq_all_enc_flock_size, self.n_frequent_seqs, self.seq_lookahead, self.n_cluster_centers),
            dtype=self._float_dtype, device=self._device)

        self.all_encountered_exploration_attempts = creator.empty(
            (freq_all_enc_flock_size, self.max_encountered_seqs, self.seq_lookahead),
            dtype=self._float_dtype, device=self._device)

        # ratio of successful transitions in each sequence
        self.all_encountered_exploration_results = creator.empty(
            (freq_all_enc_flock_size, self.max_encountered_seqs, self.seq_lookahead, self.n_cluster_centers),
            dtype=self._float_dtype, device=self._device)

        self.all_encountered_rewards_punishments = creator.empty(
            (freq_all_enc_flock_size, self.max_encountered_seqs, self.seq_lookahead, 2),
            dtype=self._float_dtype, device=self._device)

        self.frequent_rewards_punishments = creator.empty(
            (freq_all_enc_flock_size, self.n_frequent_seqs, self.seq_lookahead, 2),
            dtype=self._float_dtype, device=self._device)

        self.projection_outputs = creator.empty((self.flock_size, self.n_cluster_centers),
                                                dtype=self._float_dtype, device=self._device)

        self.action_outputs = creator.empty((self.flock_size, self.n_cluster_centers),
                                           dtype=self._float_dtype, device=self._device)

        self.action_rewards = creator.empty((self.flock_size, self.n_cluster_centers),
                                            dtype=self._float_dtype, device=self._device)

        self.action_punishments = creator.empty((self.flock_size, self.n_cluster_centers),
                                                dtype=self._float_dtype, device=self._device)

        self.input_context = creator.empty((self.flock_size, self.n_providers, NUMBER_OF_CONTEXT_TYPES, self.context_size),
                                           dtype=self._float_dtype, device=self._device)

        self.input_rewards = creator.empty((self.flock_size, 2), dtype=self._float_dtype, device=self._device)

        self.has_trained = creator.empty((self.flock_size,), dtype=creator.uint8, device=self._device)

        self.frustration = creator.empty((self.flock_size,), dtype=creator.int64, device=self._device)

        self.passive_predicted_clusters_outputs = creator.empty(
            (self.flock_size, self.seq_length, self.n_cluster_centers),
            dtype=self._float_dtype, device=self._device)

        # How many times did the temporal pooler forward and learning process run
        self.execution_counter_forward = creator.empty((self.flock_size, 1), device=self._device, dtype=creator.int64)
        self.execution_counter_learning = creator.empty((self.flock_size, 1), device=self._device, dtype=creator.int64)

        self.seq_likelihoods_by_context = creator.empty(self.flock_size, self.n_frequent_seqs, self.n_providers,
                                                      self.context_size, dtype=self._float_dtype, device=self._device)
        self.best_matching_context = creator.empty((self.flock_size, self.n_providers, self.context_size),
                                            dtype=self._float_dtype, device=self._device)

        self._init_tensor_values()
        self.context_size_checked = False

    def get_frequent_all_enc_flock_size(self):
        """Get the correct flock_size value for all_encoutered_ and frequent_ tensors.

        For this TP, it should be the same as the passed in flock_size
        """
        return self.flock_size

    def validate_params(self, params: ExpertParams):
        self._validate_universal_params(params)
        self._validate_conv_learning_params(params)

    def _validate_universal_params(self, params: ExpertParams):
        validate_predicate(lambda: params.flock_size > 0)
        validate_predicate(lambda: params.n_cluster_centers > 0)

        temporal = params.temporal
        validate_predicate(lambda: temporal.buffer_size > 0)
        validate_predicate(lambda: temporal.batch_size > 0)
        validate_predicate(lambda: temporal.learning_period > 0)
        validate_predicate(lambda: temporal.seq_length >= 2)
        validate_predicate(lambda: temporal.seq_lookahead > 0)
        validate_predicate(lambda: temporal.seq_lookbehind > 0)
        validate_predicate(lambda: temporal.n_frequent_seqs > 0)
        validate_predicate(lambda: temporal.max_encountered_seqs > 0)
        validate_predicate(lambda: temporal.forgetting_limit >= 1,
                           f"forgetting_limit {{{temporal.forgetting_limit}}} should be >= 1 to avoid too small numbers")
        validate_predicate(lambda: temporal.context_prior >= SMALL_CONSTANT)
        validate_predicate(lambda: temporal.exploration_attempts_prior >= SMALL_CONSTANT)
        validate_predicate(lambda: 0 <= temporal.exploration_probability <= 1)
        validate_predicate(lambda: temporal.incoming_context_size > 0)
        validate_predicate(lambda: 0 <= temporal.own_rewards_weight)
        validate_predicate(lambda: temporal.n_providers > 0, f"There should be at least one parent per flock.")

        validate_predicate(lambda: temporal.seq_length == temporal.seq_lookbehind + temporal.seq_lookahead)

        validate_predicate(lambda: temporal.buffer_size >= temporal.batch_size,
                           "Batch size cannot be larger than buffer size.")

        validate_predicate(lambda: temporal.max_encountered_seqs > temporal.batch_size - (temporal.seq_length - 1),
                           f"The whole bottom part of max_encountered_seqs {{{temporal.max_encountered_seqs}}} is "
                           f"rewritten by batch_size {{{temporal.batch_size}}} - "
                           f"(seq_length {{{temporal.seq_length}}} - 1),"
                           f" so there should be enough space to store the actual seqs.")

        validate_predicate(lambda: temporal.n_frequent_seqs <= temporal.max_encountered_seqs,
                           f"Frequent sequences are sampled from the all_encountered_seqs, so it should hold that "
                           f"n_frequent_seqs {{{temporal.n_frequent_seqs}}} <="
                           f" max_encountered_seqs {{{temporal.max_encountered_seqs}}}.")

        validate_predicate(lambda: temporal.n_subbatches > 0, f"The value for n_subbatches{{{temporal.n_subbatches}}} "
        f"Should be at least 1. A value of 1 has no subbatching, and any higher processes "
        f" all_encountered_sequences in a parallel fashion.")

    def _validate_conv_learning_params(self, params: ExpertParams):
        """Validation of the convSP params when seen as one expert == common."""
        temporal = params.temporal

        validate_predicate(lambda: 0 < temporal.max_new_seqs <= (temporal.batch_size - (temporal.seq_length - 1)),
                           f"Max new sequences is a limit on the number of new sequences that can be added in a single "
                           f"learning step. This cannot be larger than the number of sequences in the batch. "
                           f"(0 < {{{temporal.max_new_seqs}}} <= ({{{temporal.batch_size}}} - ({{{temporal.seq_length}}} - 1)).")

    def reset_learnt_sequences(self):
        self._init_tensor_values()

    def _init_tensor_values(self):
        self.all_encountered_seqs.fill_(-1)
        self.all_encountered_seq_occurrences.fill_(0)
        self.frequent_seqs.fill_(-1)
        self.frequent_seq_occurrences.fill_(0)
        self.frequent_context_likelihoods.fill_(0)
        self.all_encountered_context_occurrences.fill_(0)
        self.frequent_exploration_attempts.fill_(0)
        self.frequent_exploration_results.fill_(0)
        self.all_encountered_exploration_attempts.fill_(self.exploration_attempts_prior)
        self.all_encountered_exploration_results.fill_(1 / self.n_cluster_centers)
        self.projection_outputs.fill_(0)
        self.action_outputs.fill_(0)
        self.action_rewards.fill_(0)
        self.action_punishments.fill_(0)
        self.input_rewards.fill_(0)
        self._fill_empty_context_whole(self.input_context)
        self.has_trained.fill_(0)
        self.passive_predicted_clusters_outputs.fill_(0)
        self.execution_counter_forward.fill_(0)
        self.execution_counter_learning.fill_(0)
        self.all_encountered_rewards_punishments.fill_(0)
        self.frequent_rewards_punishments.fill_(0)
        self.frustration.fill_(0)

        # Delete processes as well (not necessary, but to ensure observers consistency)
        if self.learn_process is not None:
            del self.learn_process
        if self.trained_forward_process is not None:
            del self.trained_forward_process

    def _fill_empty_context_whole(self, input_context: torch.Tensor):
        input_context.fill_(0)
        self._fill_empty_context_part(input_context)
        TPFlock._fill_empty_reward_part(input_context)

    @staticmethod
    def _fill_empty_reward_part(input_context: torch.Tensor):
        """Set the current context values.

        Set rewards and punishments to 0.
        """
        input_context[:, :, 1:, :] = 0

    def _fill_empty_context_part(self, input_context_with_rewards: torch.Tensor):
        """Set the current context values (not reward/punishment part) to 1/n_cluster_centers"""
        input_context_with_rewards[:, :, 0, :] = 1 / self.n_cluster_centers

    def forward_learn(self,
                      input_clusters: torch.Tensor,
                      input_context: torch.Tensor = None,
                      input_rewards: torch.Tensor = None,
                      sp_mask: torch.Tensor = None):
        forward_mask = self._forward(input_clusters, input_context, input_rewards, sp_mask)
        if self.enable_learning:
            self._learn(forward_mask)

    # region Forward and Backward Pass

    def _every_step_computations(self):
        """Here are the computations which are done for all experts and potentially every step irrespective of the fact
        if the expert is computing this step or not."""
        # decay the old TP output projection values (experts with run will update their output values in the process)
        if self._params.temporal.output_projection_persistence != 1.0:
            self.projection_outputs.mul_(self._params.temporal.output_projection_persistence)

    def _forward(self, input_clusters, input_context, input_rewards, sp_mask):
        self._verify_context_and_rewards(input_context, input_rewards)

        self._every_step_computations()

        trained_forward_indices, untrained_forward_indices, common_mask = self._determine_forward_pass(input_clusters, sp_mask)
        # Create and run trained
        if self.trained_forward_process is not None:
            # Free memory of old process
            del self.trained_forward_process

        # TODO this causes blinking of process observers in UI. Solve it e.g. by caching the last value in observer.
        # Do not assign to self to prevent observer to fetch invalid data
        trained_process = self._trained_forward_process_factory.create(self,
                                                                       input_clusters,
                                                                       self.input_context,
                                                                       self.input_rewards,
                                                                       trained_forward_indices,
                                                                       self._device)
        self._run_process(trained_process)

        if trained_process is not None:
            self.trained_forward_process = trained_process

        untrained_process = self._untrained_forward_process_factory.create(self, input_clusters,
                                                                           self.input_context,
                                                                           self.input_rewards,
                                                                           untrained_forward_indices,
                                                                           self._device)
        self._run_process(untrained_process)

        # Find all TPs who are annoyed (i.e over the frustration threshold) and generate a random action for them
        # Also: activate their exploration bits in the buffer, so as to learn that this action they have currently
        # taken doesn't do anything
        annoyed = self.frustration > self.frustration_threshold
        annoyed_experts = annoyed.sum()

        action_ids = torch.randint(high=self.n_cluster_centers, size=(annoyed_experts.item(),), dtype=torch.int64, device=self._device)
        actions = id_to_one_hot(action_ids, vector_len=self.n_cluster_centers)

        self.action_rewards[annoyed, :] = actions * EXPLORATION_REWARD
        self.action_outputs[annoyed, :] = actions

        if annoyed_experts > 0:
            annoyed_indices = annoyed.nonzero().view(-1)

            self.buffer.set_flock_indices(annoyed_indices)
            self.buffer.exploring.store(torch.ones((annoyed_indices.numel(),), dtype=self._float_dtype, device=self._device))
            self.buffer.unset_flock_indices()

        return common_mask

    def _verify_context_and_rewards(self, input_context: torch.Tensor, input_rewards: torch.tensor):

        # Check this only once as the context and reward shapes shouldn't change during the run
        if not self.context_size_checked and input_context is not None:
            validate_predicate(lambda: input_context.shape == (self.flock_size, self.n_providers, NUMBER_OF_CONTEXT_TYPES, self._params.temporal.incoming_context_size))
        if not self.context_size_checked and input_rewards is not None:
            valid_shapes = [(2,), (1,), (self.flock_size, 2)]
            validate_predicate(lambda: input_rewards.shape in valid_shapes)

        self.context_size_checked = True

        if input_context is not None:
            self.input_context.copy_(input_context)
        else:
            self._fill_empty_reward_part(self.input_context)

        if input_rewards is not None:
            # Assemble rewards if needed
            if input_rewards.shape == (self.flock_size, 2):
                self.input_rewards.copy_(input_rewards)
            elif input_rewards.shape == (2,):
                self.input_rewards.copy_(input_rewards.unsqueeze(dim=0).expand(self.flock_size, 2))
            else:
                # This accepts positive and negative values and places the value in the correct spot
                r = torch.zeros((2,), dtype=self._float_dtype, device=self._device)
                if (input_rewards >= 0).all():
                    ind = 0
                else:
                    ind = 1
                r[ind] = torch.abs(input_rewards)[0]
                self.input_rewards.copy_(r.unsqueeze(dim=0).expand(self.flock_size, 2))
        else:
            self.input_rewards.fill_(0)

    def _determine_forward_pass(self, input: torch.Tensor, sp_mask: torch.Tensor=None):
        # run if it is the first step or the most probable cluster in the current data is different than the one
        # from the last step TP ran
        common_mask = (self.buffer.clusters.compare_with_last_data(input, detect_qualitative_difference) +
                       (self.buffer.total_data_written == 0))

        trained_mask = common_mask * self.has_trained
        untrained_mask = common_mask * negate(self.has_trained)

        # Frustrate all those TPs who are have SPs which are running, but they themselves are not running
        frustration_mask = negate(common_mask)
        if sp_mask is not None:
            frustration_mask *= sp_mask.view(-1)

        self.frustration[frustration_mask] += 1

        # Soothe frustration of TPs who are running
        self.frustration[trained_mask] = 0

        return trained_mask.nonzero(), untrained_mask.nonzero(), common_mask

    # endregion

    # region Learning

    def _learn(self, forward_mask):
        if self.learn_process is not None:
            del self.learn_process
        learning_process = self._determine_learning_process(forward_mask)
        self._run_process(learning_process)

        if learning_process is not None:
            self.learn_process = learning_process

    def _determine_learning(self, forward_mask):
        learning_period_condition = self.buffer.check_enough_new_data(self.learning_period)
        enough_data_in_buffer_condition = self.buffer.can_sample_batch(self.batch_size)

        return learning_period_condition * enough_data_in_buffer_condition * forward_mask

    def _determine_learning_process(self, forward_mask):
        # Start learning
        indices = self._determine_learning(forward_mask).nonzero()
        return self._create_learning_process(indices)

    def _create_learning_process(self, indices: torch.Tensor):
        # Only learn if there is an expert that should learn
        if indices.size(0) == 0:
            # No learning happening.
            return None

        do_subflocking = indices.size(0) != self.flock_size

        self.has_trained[indices] = 1

        learning_process = self._learning_process_class(indices,
                                                        do_subflocking,
                                                        self.buffer,
                                                        self.all_encountered_seqs,
                                                        self.all_encountered_seq_occurrences,
                                                        self.all_encountered_context_occurrences,
                                                        self.all_encountered_rewards_punishments,
                                                        self.all_encountered_exploration_attempts,
                                                        self.all_encountered_exploration_results,
                                                        self.frequent_seqs,
                                                        self.frequent_seq_occurrences,
                                                        self.frequent_context_likelihoods,
                                                        self.frequent_rewards_punishments,
                                                        self.frequent_exploration_attempts,
                                                        self.frequent_exploration_results,
                                                        self.execution_counter_learning,
                                                        self.max_encountered_seqs,
                                                        self.max_new_seqs,
                                                        self.n_frequent_seqs,
                                                        self.seq_length,
                                                        self.seq_lookahead,
                                                        self.seq_lookbehind,
                                                        self.n_cluster_centers,
                                                        self.batch_size,
                                                        self.forgetting_limit,
                                                        self.context_size,
                                                        self._context_prior,
                                                        self.exploration_attempts_prior,
                                                        self.n_subbatches,
                                                        self.n_providers,
                                                        self._device)

        return learning_process

    # endregion

    def inverse_projection(self, data: torch.Tensor, n_top_sequences: int = 1) -> torch.Tensor:
        """Calculates the inverse projection for the given output tensor.

        Output projection is computed for all frequent_seq, top n_top_sequences best matching are aggregated and
        projected to SP input space.

        Args:
            data: Tensor matching the shape of projection_output (flock_size, n_cluster_centers).
            n_top_sequences: Number of top sequences to aggregate
        """
        if data.shape != self.projection_outputs.shape:
            raise IllegalArgumentException(f"The provided tensor {list(data.shape)} doesn't match "
                                           f"the shape of projection_outputs {list(self.projection_outputs.shape)}")

        # Compute output projections for each sequence from frequent_seqs
        # [flock_size, n_cluster_centers]
        projection_outputs = torch.empty_like(self.projection_outputs)
        tp_output_projection = TPOutputProjection(self.flock_size, self.n_frequent_seqs, self.n_cluster_centers,
                                                  self.seq_length, self.seq_lookahead, self._device)
        tp_output_projection.compute_output_projection_per_sequence(self.frequent_seqs, projection_outputs)

        # Compute similarities with input data
        # [flock_size, n_frequent_seqs]
        similarities = tp_output_projection.compute_similarity(data, projection_outputs)

        # Scale similarities by seq likelihood
        similarities.mul_(self.frequent_seq_likelihoods_priors_clusters_context)

        # Take just top n_top_sequences best matching sequences
        # [flock_size, n_top_sequences]
        sorted_idxs = similarities.sort(dim=1, descending=True)[1][:, 0:n_top_sequences]
        # [flock_size, n_top_sequences, seq_length]
        indices = sorted_idxs.unsqueeze(-1).expand((self.flock_size, n_top_sequences, self.seq_length))
        # [flock_size, n_top_sequences, seq_length]
        matched_sequences = torch.gather(self.frequent_seqs, 1, indices)

        # Convert sequences to SP output space - one_hot representation
        # [flock_size, n_top_sequences * seq_length, n_cluster_centers]
        one_hots_per_flock = safe_id_to_one_hot(matched_sequences.view((self.flock_size, -1)), self.n_cluster_centers)

        # Final aggregation of sequences - just sum and normalize
        # [flock_size, n_cluster_centers]
        summed = one_hots_per_flock.sum(dim=1)
        normalize_probs_(summed, dim=1)
        return summed


class ConvTPFlock(TPFlock):
    def __init__(self,
                 params: ExpertParams,
                 creator: TensorCreator = torch,
                 trained_forward_factory=None,
                 untrained_forward_factory=None):
        super().__init__(params, creator, trained_forward_factory, untrained_forward_factory)

        # We should expand the all and frequent tensors to
        self._trained_forward_process_factory = trained_forward_factory or TrainedForwardProcessFactory(convolutional=True)
        self._learning_process_class = ConvTPFlockLearning

    def _validate_conv_learning_params(self, params: ExpertParams):
        """Validation of the convSP params when seen as one expert == common."""
        temporal = params.temporal

        combined_batch_size = params.flock_size * (temporal.batch_size + (0 if temporal.n_subbatches == 1 else temporal.seq_length - 1))

        validate_predicate(lambda: temporal.max_new_seqs > 0 and temporal.max_new_seqs <= (combined_batch_size - (temporal.seq_length - 1)),
                           f"Max new sequences is a limit on the number of new sequences that can be added in a single "
                           f"learning step. This cannot be larger than the number of sequences in a batch from every expert. "
                           f"(0 < {{{temporal.max_new_seqs}}} <= ({{{combined_batch_size}}} - ({{{temporal.seq_length}}} - 1)).")

    def get_frequent_all_enc_flock_size(self):
        """Get the correct flock_size value for all_encoutered_ and frequent_ tensors.

        For this TP, it should be 1
        """
        return 1
