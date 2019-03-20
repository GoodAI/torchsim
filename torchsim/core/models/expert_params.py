import copy
import logging
from dataclasses import dataclass, is_dataclass, fields, field
from enum import Enum

logger = logging.getLogger()

# TODO (Feat): Use yaml-able params for e.g. tests or repeated experiments.

DEFAULT_CONTEXT_PRIOR: float = 5.0  # Prior on how many times was each cluster in each sequence seen in each context
NUMBER_OF_CONTEXT_TYPES: int = 3  # Current, reward, and punishment contexts
DEFAULT_EXPLORATION_ATTEMPTS_PRIOR: float = 5.0  # Prior on how many times was each sequence explored.
EXPLORATION_REWARD = 10000  # Reward that a parent will promise a child if it the parent decides to explore.
REWARD_DISCOUNT_FACTOR = 0.9  # What factor we multiply the rewards by in order to discount them through time.


class ParamsBase:
    __isfrozen: bool = False

    def clone(self):
        return copy.deepcopy(self)

    def __setattr__(self, key, value):
        def raise_error():
            raise AttributeError(f'Attribute {key!r} not found in {self.__class__.__name__}')

        if is_dataclass(self):
            # Check fields in dataclass - uninitialized fields are not in __dict__ yet
            # noinspection PyDataclass
            if key not in [f.name for f in fields(self)] and not hasattr(self, key):
                raise_error()
        elif not hasattr(self, key):
            # Plain object
            raise_error()
        object.__setattr__(self, key, value)


class SamplingMethod(Enum):
    LAST_N = 1
    UNIFORM = 2
    BALANCED = 3


@dataclass
class SpatialPoolerParams(ParamsBase):
    # Note: when referring to steps, it means number of runs of SP forward process
    input_size: int = 256  # Size of input vector for one expert
    buffer_size: int = 2000  # Size of the SP buffer - how many last entries (steps) are stored
    batch_size: int = 200  # Size of the SP batch - it is sampled from the buffer
    learning_rate: float = 0.1  # How much of a distance between the current position of the cluster center and its
    # target position is removed in one learning process run
    cluster_boost_threshold: int = 1000  # If the cluster is without any datapoint for this many consecutive steps, the
    # boosting starts
    max_boost_time: int = 2000  # Is any cluster is boosted for this many steps, the boosting targets are recomputed
    learning_period: int = 10  # How often is the learning process run - every Xth of SP forward process runs
    enable_learning: bool = True  # If false, the SP will not learn.
    boost: bool = True  # If false, the SP will not boost clusters which have no datapoints
    sampling_method: SamplingMethod = SamplingMethod.BALANCED  # LAST_N - take last n entries from the buffer,
    #  UNIFORM - sample uniformly from the whole buffer,
    # BALANCED - sample from the whole buffer so that the counts of points belonging to each cluster are approximately
    # equal


@dataclass
class TemporalPoolerParams(ParamsBase):
    # Note: when referring to steps, it means number of runs of TP forward process
    buffer_size: int = 300  # Size of the TP buffer - how many consecutive steps are stored
    batch_size: int = 200  # How large is the batch 'sampled' from the buffer - in the case of TP the batch always
    # contains last X entries
    learning_period: int = 100  # How often does the learning of TP run (every Xth step of the TP)
    seq_length: int = 4  # length of the sequences considered in the TP, it equals lookbehind + lookahead
    seq_lookahead: int = 2  # How large part of the sequence is lookahead (rest is lookbehind including the current
    # cluster) - it needs to be at least 2 for the goal directed inference to work.
    n_frequent_seqs: int = 200  # How many of the sequences from max_encountered_seqs are used in the forward and
    # backward processes. Only X most frequent ones.
    max_encountered_seqs: int = 2000  # How many sequences does the TP know. Their statistics are updated during
    # learning. If TP encounters more sequences, if forgets the least frequent ones.
    forgetting_limit: int = 5000  # Value influencing how fast is the old knowledge in TP replaced by the new knowledge.
    # When adding new knowledge, it compresses old knowledge into X steps. This corresponds to exponential decay with
    # factor 1/X.
    context_prior: float = DEFAULT_CONTEXT_PRIOR  # What is the prior probability of seeing any new sequence in any
    # context. This eliminates too extreme judgments based on only few data. It should not be normally changed.
    exploration_attempts_prior: float = DEFAULT_EXPLORATION_ATTEMPTS_PRIOR  # Similar to the context_prior, but for
    # exploration.
    exploration_probability: float = 0.01  # With this probability, the expert will be exploring instead of trying to fulfill
    #  goals.
    follow_goals: bool = True  # If true, tries to follow goals and get as much reward as possible
    enable_learning: bool = True  # if false, the TP will not learn.
    # The maximum number of new sequences the TP will learn each learning step. It will learn the most occurring new sequences first.
    _max_new_seqs: int = None
    # How any subbatches that the TP will create from it's learning batch to speed up the processing of identifying sequences.
    # Higher number = faster, but higher memory overhead
    n_subbatches: int = 3
    # before computing new output projection, multiply the old values with this number
    output_projection_persistence: float = 1.0  # weights between fully event-driven and fully non-persistent outputs
    # Number of providers of context for this expert
    n_providers: int = 1
    # The maximum amount of context that this expert will receive (corresponds to 2 * n_cluster_centers). This value
    # Should be the same as the context sizes of all the co-parents of this flock
    incoming_context_size: int = 100
    # How highly does an expert weight its own rewards with respect to the promised rewards from the parents.
    own_rewards_weight: float = 0.1
    # How many steps a TP that could run will wait before trying something random
    frustration_threshold: int = 10
    # Of those exploring this turn. What percentage should try random clusters instead of following sequences?
    cluster_exploration_prob: float = 0.2
    # Should the active inference (goal-directed behavior, actions) be computed. If not needed, disabling this can speed
    # up the computation.
    compute_backward_pass: bool = False
    # When set to true, internal predicted_clusters_by_context and output best_matching_context are computed
    compute_best_matching_context: bool = False

    @property
    def seq_lookbehind(self):
        return self.seq_length - self.seq_lookahead

    @property
    def max_new_seqs(self):
        return self._max_new_seqs or self.batch_size - (self.seq_length - 1)

    @max_new_seqs.setter
    def max_new_seqs(self, value):
        self._max_new_seqs = value


@dataclass
class ExpertParams(ParamsBase):
    flock_size: int = 10  # how many expert in the flock
    n_cluster_centers: int = 20  # how many cluster centers does each expert have
    spatial: SpatialPoolerParams = field(default_factory=SpatialPoolerParams)  # params for the SpatialPooler only
    temporal: TemporalPoolerParams = field(default_factory=TemporalPoolerParams)  # params for the TemporalPooler only
    compute_reconstruction: bool = False  # if True reconstructions in SP will be computed, otherwise not
    produce_actions: bool = False  # if True, this flock is on the bottom level of the hirearchy and should produce actions
