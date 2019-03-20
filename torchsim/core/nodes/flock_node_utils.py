from typing import Optional, Callable, cast, Dict

from torchsim.core.models.flock import Process
from torchsim.core.models.spatial_pooler import SPFlockLearning
from torchsim.core.models.temporal_pooler import TPFlockForwardAndBackward, TPFlockLearning
from torchsim.gui.observers.flock_process_observable import FlockProcessObservable


def create_observables(definitions, flock_size: int, process_getter: Callable[[], Optional[Process]]) \
        -> Dict[str, FlockProcessObservable]:
    return {name: FlockProcessObservable(flock_size, process_getter, tensor_getter)
            for name, tensor_getter in definitions.items()}


def create_sp_learn_observables(flock_size: int, process_getter: Callable[[], Optional[SPFlockLearning]]) \
        -> Dict[str, FlockProcessObservable]:
    return create_observables({
        'SP Learn Process.Data Batch':
            lambda p: cast(SPFlockLearning, p).data_batch,
        'SP Learn Process.Last Batch Clusters':
            lambda p: cast(SPFlockLearning, p).last_batch_clusters,
        'SP Learn Process.Boost Deltas':
            lambda p: cast(SPFlockLearning, p).boost_deltas,
        'SP Learn Process.Temp. Boosting Targets':
            lambda p: cast(SPFlockLearning, p).tmp_boosting_targets
    }, flock_size, process_getter)


def create_tp_learn_observables(flock_size: int, process_getter: Callable[[], Optional[TPFlockLearning]]) \
        -> Dict[str, FlockProcessObservable]:
    return create_observables({
        'TP Learn Process.Encountered Batch Seq. Occurences':
            lambda p: cast(TPFlockLearning, p).encountered_batch_seq_occurrences,
        'TP Learn Process.Encountered Batch Context Occurences':
            lambda p: cast(TPFlockLearning, p).encountered_batch_context_occurrences,
        'TP Learn Process.Encountered Batch Exploration Attempts':
            lambda p: cast(TPFlockLearning, p).encountered_batch_exploration_attempts,
        'TP Learn Process.Encountered Batch Exploration Results':
            lambda p: cast(TPFlockLearning, p).encountered_batch_exploration_results,
        'TP Learn Process.Newly Encountered Seqs Indicator':
            lambda p: cast(TPFlockLearning, p).newly_encountered_seqs_indicator,
        'TP Learn Process.Newly Encountered Seqs Counts':
            lambda p: cast(TPFlockLearning, p).newly_encountered_seqs_counts,
        'TP Learn Process.Most Probable Batch Seqs':
            lambda p: cast(TPFlockLearning, p).most_probable_batch_seqs,
        'TP Learn Process.Most Probable Batch Seq Probs':
            lambda p: cast(TPFlockLearning, p).most_probable_batch_seq_probs,
        'TP Learn Process.Total Encountered Occurences':
            lambda p: cast(TPFlockLearning, p).total_encountered_occurrences,
        'TP Learn Process.Cluster Batch':
            lambda p: cast(TPFlockLearning, p).cluster_batch,
        'TP Learn Process.Context Batch':
            lambda p: cast(TPFlockLearning, p).context_batch,
        'TP Learn Process.Exploring Batch':
            lambda p: cast(TPFlockLearning, p).exploring_batch,
        'TP Learn Process.Actions Batch':
            lambda p: cast(TPFlockLearning, p).actions_batch
    }, flock_size, process_getter)


def create_tp_forward_observables(flock_size: int, process_getter: Callable[[], Optional[TPFlockForwardAndBackward]]) \
        -> Dict[str, FlockProcessObservable]:
    return create_observables({
        'TP Forward Process.Cluster History':
            lambda p: cast(TPFlockForwardAndBackward, p).cluster_history,
        'TP Forward Process.Context History':
            lambda p: cast(TPFlockForwardAndBackward, p).context_history,
        # ['flock_size', 'tp_n_frequent_seqs'] -> 'likelihood'
        # Likelihood that TP is in given sequence based just on cluster history of size tp_n_seq_lookbehind
        'TP Forward Process.Seq. Likelihoods Clusters':
            lambda p: cast(TPFlockForwardAndBackward, p).seq_likelihoods_clusters,
        # ['flock_size', 'tp_n_frequent_seqs'] -> 'likelihood'
        # Not normalized likelihood that TP is in given sequence based on cluster history of size tp_n_seq_lookbehind
        # and frequency of the sequence. It's tp_forward_process_seq_likelihoods_clusters * tp_frequent_seq_occurrences
        'TP Forward Process.Seq. Likelihoods Priors Clusters':
            lambda p: cast(TPFlockForwardAndBackward, p).seq_likelihoods_priors_clusters,
        'TP Forward Process.Seq. Likelihoods For Each Provider':
            lambda p: cast(TPFlockForwardAndBackward, p).seq_likelihoods_for_each_provider,
        'TP Forward Process.Seq. Likelihoods Priors Clusters Context':
            lambda p: cast(TPFlockForwardAndBackward, p).seq_likelihoods_priors_clusters_context,
        # 'TP Forward Process.Seq. Likelihoods By Context':
        #     lambda p: cast(TPFlockForwardAndBackward, p).seq_likelihoods_by_context,
        'TP Forward Process.predicted_clusters_by_context':
            lambda p: cast(TPFlockForwardAndBackward, p).predicted_clusters_by_context,
        # ['flock_size', 'tp_n_frequent_seqs'] -> 'probability'
        # Probability of the sequence in the most informative context.
        'TP Forward Process.Seq. Probs Clusters Context':
            lambda p: cast(TPFlockForwardAndBackward, p).seq_probs_clusters_context,
        'TP Forward Process.Seq. Likelihoods Exploration':
            lambda p: cast(TPFlockForwardAndBackward, p).seq_likelihoods_exploration,
        'TP Forward Process.Seq. Likelihoods Active':
            lambda p: cast(TPFlockForwardAndBackward, p).seq_likelihoods_active,
        'TP Forward Process.Active Predicted Clusters':
            lambda p: cast(TPFlockForwardAndBackward, p).active_predicted_clusters,
        'TP Forward Process.Exploring':
            lambda p: cast(TPFlockForwardAndBackward, p).exploring,
        'TP Forward Process.Exploration Random Numbers':
            lambda p: cast(TPFlockForwardAndBackward, p).exploration_random_numbers,
        'TP Forward Process.Context Informativeness':
            lambda p: cast(TPFlockForwardAndBackward, p).context_informativeness,
        'TP Forward Process.Frequent Seqs Scaled':
            lambda p: cast(TPFlockForwardAndBackward, p).frequent_seqs_scaled,
        'TP Forward Process.Seq. Likelihoods Goal Directed':
            lambda p: cast(TPFlockForwardAndBackward, p).seq_rewards_goal_directed
    }, flock_size, process_getter)
