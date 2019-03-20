#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <kernel_helpers.h>

using namespace at;

/*
        Counts known sequence occurrences (for each sequence from all encountered
        Remove marks at the starts of new sequences (if the sequence is present in all encountered and
        have probability > 0.5) next to their first clusters
        ( Replaces 1's in newly_encountered_seqs with 0's if the seq starting
        at that batch position is in all_encountered_seqs)
*/
template <typename scalar_t>
__global__ void count_batch_seq_occurrences_cuda_kernel(
    const TensorIndexer<scalar_t, 4> cluster_batch,  // input [flock_size, n_subbatches, subbatch_size, n_cluster_centers]
    const TensorIndexer<scalar_t, 5> context_batch,  // input [flock_size, n_subbatches, subbatch_size, n_providers, context_size]
    const TensorIndexer<scalar_t, 4> rewards_punishment_batch, // input [flock_size, n_subbatches, subbatch_size, 2]
    const TensorIndexer<int64_t, 3> all_encountered_seqs,  // input [flock_size, max_encountered_seqs, seq_length]
    const TensorIndexer<scalar_t, 2> all_encountered_seq_occurrences,  // input [flock_size, max_encountered_seqs]
    TensorIndexer<scalar_t, 3> encountered_batch_seq_occurrences,  // output [flock_size, n_subbatches, max_encountered_seqs]
    TensorIndexer<scalar_t, 6> encountered_batch_context_occurrences, // output [flock_size, n_subbatches, max_encountered_seqs, seq_length, n_providers, context_size]
    TensorIndexer<scalar_t, 5> encountered_batch_rewards_punishments, // output [flock_size, n_subbatches, max_encountered_seqs, seq_lookahead, 2]
    TensorIndexer<int64_t, 2> newly_encountered_seqs_indicator,  // input/output [flock_size, n_subbatches * subbatch_size]
    //TensorIndexer<int64_t, 3> newly_encountered_seqs_indicator,  // input/output [flock_size, n_subbatches, subbatch_size]
    const TensorIndexer<scalar_t, 4> explorations,  // input [flock_size, n_subbatches, subbatch_size, 1 (the singular dimension needs
     //to be here, see TPFlockBuffer)]
    const  TensorIndexer<scalar_t, 4> actions,  // input [flock_size, n_subbatches, subbatch_size, n_cluster_centers]
    TensorIndexer<scalar_t, 4> encountered_exploration_attempts,  // output [flock_size, n_subbatches, max_encountered_seqs, seq_lookahead]
    TensorIndexer<scalar_t, 5> encountered_exploration_results,  // output [flock_size, n_subbatches, max_encountered_seqs, seq_lookahead, n_cluster_centers]
    int n_cluster_centers,
    int seq_length,
    int seq_lookbehind,
    int used_encountered_sequences,  // the number which is used (all minus the max_new_seq_stored)
    int max_seqs_in_subbatch,
    int context_size,
    int n_subbatches,
    int max_seqs_in_batch,
    int n_providers,
    int required_threads)
{
  const int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < required_threads)
  {
    int encountered_seq_idx = id % used_encountered_sequences; 
    int subbatch_idx = (id / used_encountered_sequences) %  n_subbatches;
    int flock_idx = id / (used_encountered_sequences * n_subbatches);

    // The sum of probability of occurrence of this thread's encountered sequence in the whole cluster_batch.
    scalar_t seq_occurrence = 0;

    // Only count sequences in the cluster_batch which were encountered before.
    scalar_t occurrences = all_encountered_seq_occurrences.at(flock_idx, encountered_seq_idx);
    if (occurrences > 0)
    {
      // Go over all positions in the cluster_batch. Calculate how much is the thread's encountered sequence
      // present at [seq_start, seq_start+1, ... seq_start + seq_length].
      for (int seq_start = 0; seq_start < max_seqs_in_subbatch; seq_start++) {
        // The probability that this sequence in the cluster_batch is this thread's encountered sequence.
        scalar_t seq_prob = 1.0;
        // Go over all clusters in the sequence and compare them with the cluster probabilities in the cluster_batch.
        for (int cluster_idx = 0; cluster_idx < seq_length; cluster_idx++) {
          int subbatch_line = seq_start + cluster_idx;
          int cluster = all_encountered_seqs.at(flock_idx, encountered_seq_idx, cluster_idx);

          scalar_t cluster_prob = cluster_batch.at(flock_idx, subbatch_idx, subbatch_line, cluster);
          seq_prob *= cluster_prob;

          int transition_idx = cluster_idx - (seq_lookbehind - 1);

          // --------------------------------  Influence model  ----------------------------------------
          //from exploration - run only if cluster_idx corresponds to a first cluster of a transition
          // in the lookahead part of the sequence and we were exploring during this transition
          if (transition_idx >= 0 && cluster_idx != seq_length - 1 && explorations.at(flock_idx, subbatch_idx, subbatch_line, 0) == 1) {
            //count this transition only if we were exploring

            // We only care about an action towards the next cluster of this sequence
            int next_cluster = all_encountered_seqs.at(flock_idx, encountered_seq_idx, cluster_idx + 1);
            scalar_t prob_attempted_cluster = actions.at(flock_idx, subbatch_idx, subbatch_line, next_cluster);

            // Work out the attempt probability and save it
            scalar_t attempts = seq_prob * prob_attempted_cluster;
            encountered_exploration_attempts.at(flock_idx, subbatch_idx, encountered_seq_idx, transition_idx) += attempts;

            // Iterate through the ending cluster probabilities, and multiply that resulting cluster by the attempt
            for(int cluster_end_idx = 0; cluster_end_idx < n_cluster_centers; cluster_end_idx++){

                scalar_t prob_ending_cluster = cluster_batch.at(flock_idx, subbatch_idx, subbatch_line + 1, cluster_end_idx);
                scalar_t successes = attempts * prob_ending_cluster;
                encountered_exploration_results.at(flock_idx, subbatch_idx, encountered_seq_idx, transition_idx, cluster_end_idx) += successes;
            }
          }


        }

        // If the sequence is more probable than 50%, mark it as not new.
        int index = subbatch_idx * max_seqs_in_subbatch + seq_start;
        if (seq_prob > 0.5 && index < max_seqs_in_batch)
          newly_encountered_seqs_indicator.at(flock_idx, index) = 0;

        seq_occurrence += seq_prob;

        // -----------------------------------  Context  ---------------------------------------------
        for (int provider_idx = 0; provider_idx < n_providers; provider_idx++){
          for (int context_idx = 0; context_idx < context_size; context_idx++){
            for (int cluster_idx = 0; cluster_idx < seq_length; cluster_idx++){
              int subbatch_line = seq_start + cluster_idx;

              scalar_t context_prob = context_batch.at(flock_idx, subbatch_idx, subbatch_line, provider_idx, context_idx);
              encountered_batch_context_occurrences.at(flock_idx, subbatch_idx, encountered_seq_idx, cluster_idx, provider_idx, context_idx) += seq_prob * context_prob;
            }
          }
        }

        // ---------------------------- Rewards Model -------------------------------------
        // We assume that if a cluster has a reward, the transition from the previous cluster was what generated
        // the reward. Therefore we add any rewards in the batch at a relevant cluster to the sequence transition
        // multiplied by the likelihood of the sequence.
        for (int cluster_idx = seq_lookbehind; cluster_idx < seq_length; cluster_idx++){
          int lookahead_transition_idx = cluster_idx - seq_lookbehind;
          int subbatch_line = seq_start + cluster_idx;

          scalar_t reward = rewards_punishment_batch.at(flock_idx, subbatch_idx, subbatch_line, 0) * seq_prob;
          scalar_t punishment = rewards_punishment_batch.at(flock_idx, subbatch_idx, subbatch_line, 1) * seq_prob;

          encountered_batch_rewards_punishments.at(flock_idx, subbatch_idx, encountered_seq_idx, lookahead_transition_idx, 0) += reward;
          encountered_batch_rewards_punishments.at(flock_idx, subbatch_idx, encountered_seq_idx, lookahead_transition_idx, 1) += punishment;

        }
      }
    }

    // Store the result of the batch sequence counting.
    encountered_batch_seq_occurrences.at(flock_idx, subbatch_idx, encountered_seq_idx) = seq_occurrence;
  }
}

void count_batch_seq_occurrences(at::Tensor cluster_batch,
                            at::Tensor context_batch,
                            at::Tensor rewards_punishment_batch,
                            at::Tensor all_encountered_seqs,
                            at::Tensor all_encountered_seq_occurrences,
                            at::Tensor encountered_batch_seq_occurrences,
                            at::Tensor encountered_batch_context_occurrences,
                            at::Tensor encountered_batch_rewards_punishments,
                            at::Tensor newly_encountered_seqs_indicator,
                            at::Tensor explorations,
                            at::Tensor actions,
                            at::Tensor encountered_exploration_attempts,
                            at::Tensor encountered_exploration_results,
                            int flock_size,
                            int n_cluster_centers,
                            int seq_length,
                            int seq_lookbehind,
                            int used_encountered_seqs,
                            int max_seqs_in_subbatch,
                            int context_size,
                            int n_subbatches,
                            int max_seqs_in_batch,
                            int n_providers)
{
    CHECK_INPUT(cluster_batch);
    CHECK_INPUT(context_batch);
    CHECK_INPUT(rewards_punishment_batch);
    CHECK_INPUT(all_encountered_seqs);
    CHECK_INPUT(all_encountered_seq_occurrences);
    CHECK_INPUT(encountered_batch_seq_occurrences);
    CHECK_INPUT(encountered_batch_context_occurrences);
    CHECK_INPUT(encountered_batch_rewards_punishments);

    CHECK_INPUT(newly_encountered_seqs_indicator);
    CHECK_INPUT(explorations);
    CHECK_INPUT(actions);
    CHECK_INPUT(encountered_exploration_attempts);
    CHECK_INPUT(encountered_exploration_results);

    const auto required_threads = flock_size * n_subbatches * used_encountered_seqs;
    const int blocks = GET_BLOCK_COUNT(required_threads);
    const cudaStream_t stream = set_device_get_cuda_stream(cluster_batch.get_device());

    AT_DISPATCH_FLOATING_TYPES(cluster_batch.type(), "count_batch_seq_occurrences_cuda_kernel", ([&] {
        count_batch_seq_occurrences_cuda_kernel<scalar_t><<<blocks, max_threads_per_block, 0, stream>>>(
        TensorIndexer<scalar_t, 4>(cluster_batch),
        TensorIndexer<scalar_t, 5>(context_batch),
        TensorIndexer<scalar_t, 4>(rewards_punishment_batch),
        TensorIndexer<int64_t, 3>(all_encountered_seqs),
        TensorIndexer<scalar_t, 2>(all_encountered_seq_occurrences),
        TensorIndexer<scalar_t, 3>(encountered_batch_seq_occurrences),
        TensorIndexer<scalar_t, 6>(encountered_batch_context_occurrences),
        TensorIndexer<scalar_t, 5>(encountered_batch_rewards_punishments),
        TensorIndexer<int64_t, 2>(newly_encountered_seqs_indicator),
        //TensorIndexer<int64_t, 3>(newly_encountered_seqs_indicator),
        TensorIndexer<scalar_t, 4>(explorations),
        TensorIndexer<scalar_t, 4>(actions),
        TensorIndexer<scalar_t, 4>(encountered_exploration_attempts),
        TensorIndexer<scalar_t, 5>(encountered_exploration_results),
        n_cluster_centers,
        seq_length,
        seq_lookbehind,
        used_encountered_seqs,
        max_seqs_in_subbatch,
        context_size,
        n_subbatches,
        max_seqs_in_batch,
        n_providers,
        required_threads);
    }));
}

