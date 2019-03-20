from typing import TYPE_CHECKING

from abc import ABC

from torchsim.core.models.expert_params import SpatialPoolerParams, ExpertParams, SamplingMethod, TemporalPoolerParams
from torchsim.core.models.spatial_pooler import SPFlock
from torchsim.core.models.temporal_pooler import TPFlock
from torchsim.gui.observables import Initializable, ObserverPropertiesBuilder, ObserverPropertiesItem, disable_on_runtime, \
    ObserverPropertiesItemSourceType
from torchsim.gui.validators import *

if TYPE_CHECKING:
    from torchsim.core.nodes.expert_node import ExpertFlockUnit


class SpatialPoolerParamsProps(Initializable, ABC):
    _flock: SPFlock
    _prop_builder: ObserverPropertiesBuilder
    _params: SpatialPoolerParams

    def __init__(self, params: SpatialPoolerParams, flock: SPFlock):
        self._flock = flock
        self._params = params
        self._prop_builder = ObserverPropertiesBuilder(self, source_type=ObserverPropertiesItemSourceType.MODEL)

    def is_initialized(self) -> bool:
        return self._flock is not None

    @property
    def input_size(self) -> int:
        return self._params.input_size

    @property
    def buffer_size(self) -> int:
        return self._params.buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int):
        validate_positive_int(value)
        if value < self.batch_size:
            raise FailedValidationException('buffer_size must be equal or greater then batch_size')
        self._params.buffer_size = value

    @property
    def batch_size(self) -> int:
        return self._params.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        validate_positive_int(value)
        if value > self.buffer_size:
            raise FailedValidationException('batch_size must be equal or less then buffer_size')
        self._params.batch_size = value

    @property
    def learning_rate(self) -> float:
        return self._params.learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        validate_positive_float(value)
        self._params.learning_rate = value
        if self._flock is not None:
            self._flock.learning_rate = value

    @property
    def cluster_boost_threshold(self) -> int:
        return self._params.cluster_boost_threshold

    @cluster_boost_threshold.setter
    def cluster_boost_threshold(self, value: int):
        validate_positive_int(value)
        self._params.cluster_boost_threshold = value

    @property
    def max_boost_time(self) -> int:
        return self._params.max_boost_time

    @max_boost_time.setter
    def max_boost_time(self, value: int):
        validate_positive_int(value)
        self._params.max_boost_time = value

    @property
    def learning_period(self) -> int:
        return self._params.learning_period

    @learning_period.setter
    def learning_period(self, value: int):
        validate_positive_int(value)
        self._params.learning_period = value

    @property
    def enable_learning(self) -> bool:
        return self._params.enable_learning

    def reset_cluster_centers(self):
        self._flock.initialize_cluster_centers()

    @enable_learning.setter
    def enable_learning(self, value: bool):
        self._params.enable_learning = value
        if self._flock is not None:
            self._flock.enable_learning = value

    @property
    def boost(self) -> bool:
        return self._params.boost

    @boost.setter
    def boost(self, value: bool):
        self._params.boost = value

    @property
    def sampling_method(self) -> SamplingMethod:
        return self._params.sampling_method

    @sampling_method.setter
    def sampling_method(self, value: SamplingMethod):
        self._params.sampling_method = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
            self._prop_builder.auto('SP_input_size', type(self).input_size, edit_strategy=disable_on_runtime,
                                    hint='Size of input vector for one expert'),
            self._prop_builder.auto('SP_buffer_size', type(self).buffer_size, edit_strategy=disable_on_runtime,
                                    hint='Size of the SP buffer - how many last entries (steps) are stored'),
            self._prop_builder.auto('SP_batch_size', type(self).batch_size, edit_strategy=disable_on_runtime,
                                    hint='Size of the SP batch - it is sampled from the buffer'),
            self._prop_builder.auto('SP_learning_rate', type(self).learning_rate,
                                    hint='How much of a distance between the current position of the cluster center '
                                         'and its target position is removed in one learning process run'),
            self._prop_builder.auto('SP_enable_learning', type(self).enable_learning, hint='SP learning is enabled'),
            self._prop_builder.button('SP_reset_cluster_centers', self.reset_cluster_centers),
            #
            self._prop_builder.auto('SP_cluster_boost_threshold', type(self).cluster_boost_threshold,
                                    edit_strategy=disable_on_runtime,
                                    hint='If the cluster is without any datapoint for this many consecutive steps, '
                                         'the boosting starts'),
            self._prop_builder.auto('SP_max_boost_time', type(self).max_boost_time, edit_strategy=disable_on_runtime,
                                    hint='Is any cluster is boosted for this many steps, the boosting targets are '
                                         'recomputed'),
            self._prop_builder.auto('SP_learning_period', type(self).learning_period, edit_strategy=disable_on_runtime,
                                    hint='How often is the learning process run - every Xth of SP forward process runs'),
            self._prop_builder.auto('SP_boost', type(self).boost, edit_strategy=disable_on_runtime,
                                    hint='If false, the SP will not boost clusters which have no datapoints'),
            self._prop_builder.auto('SP_sampling_method', type(self).sampling_method, edit_strategy=disable_on_runtime,
                                    hint='<ul>'
                                         '<li>LAST_N - take last n entries from the buffer</li>'
                                         '<li>UNIFORM - sample uniformly from the whole buffer</li>'
                                         '<li>BALANCED - sample from the whole buffer so that the counts of points '
                                         'belonging to each cluster are approximately equal</li>'
                                         '</ul>'),
        ]


class TemporalPoolerParamsProps(Initializable, ABC):
    _flock: TPFlock
    _prop_builder: ObserverPropertiesBuilder
    _params: TemporalPoolerParams

    def __init__(self, params: TemporalPoolerParams, flock: TPFlock):
        self._flock = flock
        self._params = params
        self._prop_builder = ObserverPropertiesBuilder(self, source_type=ObserverPropertiesItemSourceType.MODEL)

    def is_initialized(self) -> bool:
        return self._flock is not None

    @property
    def own_rewards_weight(self) -> float:
        return self._params.own_rewards_weight

    @own_rewards_weight.setter
    def own_rewards_weight(self, value: float):
        validate_positive_with_zero_float(value)
        self._params.own_rewards_weight = value

    @property
    def incoming_context_size(self) -> int:
        return self._params.incoming_context_size

    @incoming_context_size.setter
    def incoming_context_size(self, value: int):
        validate_positive_with_zero_int(value)
        self._params.incoming_context_size = value

    @property
    def buffer_size(self) -> int:
        return self._params.buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int):
        validate_positive_int(value)
        if value < self.batch_size:
            raise FailedValidationException('buffer_size must be equal or greater then batch_size')
        self._params.buffer_size = value

    @property
    def batch_size(self) -> int:
        return self._params.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        validate_positive_int(value)
        if value > self.buffer_size:
            raise FailedValidationException('batch_size must be equal or less then buffer_size')
        self._params.batch_size = value

    @property
    def learning_period(self) -> int:
        return self._params.learning_period

    @learning_period.setter
    def learning_period(self, value: int):
        validate_positive_int(value)
        self._params.learning_period = value
        if self._flock is not None:
            self._flock.learning_period = value

    @property
    def enable_learning(self) -> bool:
        return self._params.enable_learning

    @enable_learning.setter
    def enable_learning(self, value: bool):
        self._params.enable_learning = value
        if self._flock is not None:
            self._flock.enable_learning = value

    @property
    def seq_length(self) -> int:
        return self._params.seq_length

    @seq_length.setter
    def seq_length(self, value: int):
        validate_positive_int(value)
        self._params.seq_length = value

    @property
    def seq_lookahead(self) -> int:
        return self._params.seq_lookahead

    @seq_lookahead.setter
    def seq_lookahead(self, value: int):
        validate_positive_int(value)
        self._params.seq_lookahead = value

    @property
    def n_frequent_seqs(self) -> int:
        return self._params.n_frequent_seqs

    @n_frequent_seqs.setter
    def n_frequent_seqs(self, value: int):
        validate_positive_int(value)
        self._params.n_frequent_seqs = value

    @property
    def max_encountered_seqs(self) -> int:
        return self._params.max_encountered_seqs

    @max_encountered_seqs.setter
    def max_encountered_seqs(self, value: int):
        validate_positive_int(value)
        self._params.max_encountered_seqs = value

    @property
    def forgetting_limit(self) -> int:
        return self._params.forgetting_limit

    @forgetting_limit.setter
    def forgetting_limit(self, value: int):
        validate_positive_int(value)
        if self.is_initialized():
            self._flock.forgetting_limit = value
        self._params.forgetting_limit = value

    @property
    def context_prior(self) -> float:
        return self._params.context_prior

    @context_prior.setter
    def context_prior(self, value: float):
        validate_positive_float(value)
        self._params.context_prior = value

    @property
    def exploration_attempts_prior(self) -> float:
        return self._params.exploration_attempts_prior

    @exploration_attempts_prior.setter
    def exploration_attempts_prior(self, value: float):
        validate_positive_float(value)
        self._params.exploration_attempts_prior = value

    @property
    def exploration_probability(self) -> float:
        return self._params.exploration_probability

    @exploration_probability.setter
    def exploration_probability(self, value: float):
        validate_positive_with_zero_float(value)
        self._params.exploration_probability = value
        if self.is_initialized():
            self._flock.exploration_probability = value

    @property
    def output_projection_persistance(self) -> float:
        return self._params.output_projection_persistence

    @output_projection_persistance.setter
    def output_projection_persistance(self, value: float):
        validate_float_in_range(value, 0, 1)
        self._params.output_projection_persistence = value

    @property
    def follow_goals(self) -> bool:
        return self._params.follow_goals

    @follow_goals.setter
    def follow_goals(self, value: bool):
        self._params.follow_goals = value
        if self.is_initialized():
            self._flock.follow_goals = value

    def reset_learnt_sequences(self):
        self._flock.reset_learnt_sequences()

    @property
    def compute_backward_pass(self) -> bool:
        return self._params.compute_backward_pass

    @compute_backward_pass.setter
    def compute_backward_pass(self, value: bool):
        self._params.compute_backward_pass = value
        if self.is_initialized():
            self._flock.compute_backward_pass = value

    @property
    def compute_best_matching_context(self) -> bool:
        return self._params.compute_best_matching_context

    @compute_best_matching_context.setter
    def compute_best_matching_context(self, value: bool):
        self._params.compute_best_matching_context = value
        if self.is_initialized():
            self._flock.compute_best_matching_context = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
            self._prop_builder.button('TP_reset_learnt_sequences', self.reset_learnt_sequences),
            self._prop_builder.auto('TP_incoming_context_size', type(self).incoming_context_size,
                                    edit_strategy=disable_on_runtime,
                                    hint='Size of the context input without the two elements for reward'),
            self._prop_builder.auto('TP_buffer_size', type(self).buffer_size, edit_strategy=disable_on_runtime,
                                    hint='Size of the TP buffer - how many consecutive steps are stored'),
            self._prop_builder.auto('TP_batch_size', type(self).batch_size, edit_strategy=disable_on_runtime,
                                    hint="How large is the batch 'sampled' from the buffer - in the case of TP "
                                         "the batch always contains last X entries"),
            self._prop_builder.auto('TP_learning_period', type(self).learning_period,
                                    hint='How often does the learning of TP run (every Xth step of the TP)'),
            self._prop_builder.auto('TP_enable_learning', type(self).enable_learning, hint='TP learning is enabled'),
            self._prop_builder.auto('TP_seq_length', type(self).seq_length, edit_strategy=disable_on_runtime,
                                    hint='Length of the sequences considered in the TP, it equals lookbehind + lookahead'),
            self._prop_builder.auto('TP_seq_lookahead', type(self).seq_lookahead, edit_strategy=disable_on_runtime,
                                    hint='How large part of the sequence is lookahead (rest is lookbehind including '
                                         'the current cluster)'),
            self._prop_builder.auto('TP_n_frequent_seqs', type(self).n_frequent_seqs, edit_strategy=disable_on_runtime,
                                    hint='How many of the sequences from max_encountered_seqs are used in the forward '
                                         'and backward processes. Only X most frequent ones.'),
            self._prop_builder.auto('TP_max_encountered_seqs', type(self).max_encountered_seqs,
                                    edit_strategy=disable_on_runtime,
                                    hint='How many sequences does the TP know. Their statistics are updated during '
                                         'learning. If TP encounters more sequences, if forgets the least frequent ones.'),
            self._prop_builder.auto('TP_forgetting_limit', type(self).forgetting_limit,
                                    hint='Value influencing how fast is the old knowledge in TP replaced by the new '
                                         'knowledge. When adding new knowledge, it compresses old knowledge into X steps. '
                                         'This corresponds to exponential decay with factor 1/X.'),
            self._prop_builder.auto('TP_context_prior', type(self).context_prior, edit_strategy=disable_on_runtime,
                                    hint='What is the prior probability of seeing any new sequence in any context. '
                                         'This eliminates too extreme judgments based on only few data. '
                                         'It should not be normally changed.'),
            self._prop_builder.auto('TP_exploration_attempts_prior', type(self).exploration_attempts_prior,
                                    edit_strategy=disable_on_runtime,
                                    hint='Similar to the context_prior, but for exploration.'),
            self._prop_builder.auto('TP_exploration_probability', type(self).exploration_probability,
                                    hint='With this probability, the expert will be exploring instead of trying to '
                                         'fulfill goals.'),
            self._prop_builder.auto('TP_follow_goals', type(self).follow_goals,
                                    hint='Should the expert fulfill the goals rather just trying to do what it '
                                         'predicts will happen (trying to actively fulfill what the passive model '
                                         'predicts). True means that it tries to fulfill the goals.'),

            self._prop_builder.auto('TP_output_projection_persistence', type(self).output_projection_persistance,
                                    hint='This decays output_projection values in time (less event-driven behavior. '
                                         'Multiply output_projection by this number, compute new values of '
                                         'output_projection for experts that changed their inputs, '
                                         'set their values in the output_projection.'),

            self._prop_builder.auto("TP_own_rewards_weight", type(self).own_rewards_weight),
            self._prop_builder.auto('TP_compute_backward_pass', type(self).compute_backward_pass,
                                    hint='Should the active inference (goal-directed behavior, actions) be computed. '
                                         'If not needed, disabling this can speed up the computation'),
            self._prop_builder.auto('TP_compute_best_matching_context', type(self).compute_best_matching_context,
                                    hint='When set to true, internal predicted_clusters_by_context and output best_matching_context are computed'),
        ]


class ExpertParamsProps(Initializable, ABC):
    _unit: 'ExpertFlockUnit'
    _prop_builder: ObserverPropertiesBuilder
    _params: ExpertParams

    def __init__(self, params: ExpertParams, unit: 'ExpertFlockUnit'):
        self._unit = unit
        self._params = params
        self._prop_builder = ObserverPropertiesBuilder(self, source_type=ObserverPropertiesItemSourceType.MODEL)

    def is_initialized(self) -> bool:
        return self._unit is not None

    @property
    def flock_size(self) -> int:
        return self._params.flock_size

    @flock_size.setter
    def flock_size(self, value: int):
        validate_positive_int(value)
        self._params.flock_size = value

    @property
    def n_cluster_centers(self) -> int:
        return self._params.n_cluster_centers

    @n_cluster_centers.setter
    def n_cluster_centers(self, value: int):
        validate_positive_int(value)
        self._params.n_cluster_centers = value

    @property
    def compute_reconstruction(self) -> bool:
        return self._params.compute_reconstruction

    @compute_reconstruction.setter
    def compute_reconstruction(self, value: bool):
        self._params.compute_reconstruction = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
            self._prop_builder.auto('flock_size', type(self).flock_size, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('n_cluster_centers', type(self).n_cluster_centers,
                                    edit_strategy=disable_on_runtime),
            self._prop_builder.auto('compute_reconstruction', type(self).compute_reconstruction,
                                    edit_strategy=disable_on_runtime),
        ]
