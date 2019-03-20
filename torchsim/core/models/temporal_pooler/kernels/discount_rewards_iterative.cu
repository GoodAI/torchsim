#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <kernel_helpers.h>

#define EPSILON 0.0001

using namespace at;


template <typename scalar_t>
__global__ void discount_rewards_iterative_cuda(
    const TensorIndexer<int64_t, 3> frequent_sequences, // input[flock_size, n_frequent_seqs, seq_len]
    const TensorIndexer<scalar_t, 2> seq_probs_priors_clusters_context, // input [flock_size, n_frequent_seqs]
    const TensorIndexer<scalar_t, 5> current_rewards,  // input [flock_size, n_frequent_seqs, seq_len, n_providers, 2]
    const TensorIndexer<scalar_t, 4> influence_model, // input [flock_size, n_frequent_seqs, seq_lookahead, n_cluster_centers]
    TensorIndexer<scalar_t, 5> cluster_rewards, // output [flock_size, n_frequent_seqs, n_providers, 2, n_cluster_centers]
    int flock_size,
    int n_frequent_seqs,
    int n_cluster_centers,
    int transition_idx,
    int n_providers,
    int required_threads)
{
  const int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < required_threads){
    int cluster_idx = id % n_cluster_centers;
    int provider_idx = (id / n_cluster_centers) % n_providers;
    int seq_idx = (id / (n_cluster_centers * n_providers)) % n_frequent_seqs;
    int flock_idx = id  / (n_frequent_seqs * n_providers * n_cluster_centers);

    // We are discounting at the point of transition_idx, so check up to transition_idx + 1 clusters behind the current one
    int clusters_to_check = transition_idx + 1;
    int target_seq_idx = -1;

    // If this frequent sequence is defined
    if(frequent_sequences.at(flock_idx, seq_idx, 0) != -1){
      //printf("id %d, cluster_idx: %d, provider_idx = %d, seq_idx = %d, flock_idx: %d\n", id, cluster_idx, provider_idx, seq_idx, flock_idx);

      // If the cluster we are looking at is not the same as our sequence, then find all sequences with
      // the same beginning as the current sequence.
      if(cluster_idx != frequent_sequences.at(flock_idx, seq_idx, transition_idx+1)){

        // Search through all frequent sequences for all those sequences which have the same beginning as our current sequence
        // With the same cluster_idx as our thread
        bool found;
        scalar_t accumulated_probs = EPSILON;
        for (int search_seq_idx = 0; search_seq_idx < n_frequent_seqs; search_seq_idx++){
          found = true;
          // Match the beginning of the sequence
          for (int cc_idx = 0; cc_idx < clusters_to_check; cc_idx++){
            if (frequent_sequences.at(flock_idx, search_seq_idx, cc_idx) != frequent_sequences.at(flock_idx, seq_idx, cc_idx)){
              found = false;
              break;
            }
          }

          // If the beginning has been matched and the current cluster is the same
          // then we have the sequence we've been searching for
          if(found && frequent_sequences.at(flock_idx, search_seq_idx, clusters_to_check) == cluster_idx){
            target_seq_idx = search_seq_idx;

            scalar_t scale = seq_probs_priors_clusters_context.at(flock_idx, target_seq_idx);
            accumulated_probs += scale;

            scalar_t reward = current_rewards.at(flock_idx, target_seq_idx, transition_idx+1, provider_idx, 0) *
                        influence_model.at(flock_idx, seq_idx, transition_idx, cluster_idx) * scale;
            scalar_t punishment = current_rewards.at(flock_idx, target_seq_idx, transition_idx+1, provider_idx, 1) *
                        influence_model.at(flock_idx, seq_idx, transition_idx, cluster_idx) * scale;

            // The reward for transitioning to this cluster is the probability of getting there when transitioning,
            // times the scaled reward. Accumulate the rewards and punishments.
            cluster_rewards.at(flock_idx, seq_idx, provider_idx, 0, cluster_idx) += reward;
            cluster_rewards.at(flock_idx, seq_idx, provider_idx, 1, cluster_idx) += punishment;
          }
        }
        // We have rewards for all possible sequences leading from this cluster scaled by their prior probs. But the
        // prior probs for this subsequence ar different from the global ones, so scale up the values by the missing
        // volume of reward/punishment.
        cluster_rewards.at(flock_idx, seq_idx, provider_idx, 0, cluster_idx) *= (1/accumulated_probs);
        cluster_rewards.at(flock_idx, seq_idx, provider_idx, 1, cluster_idx) *= (1/accumulated_probs);
      }else{

        // If the cluster we are looking at is the next cluster in our sequence, then there will be exactly one
        // reward and we won't scale these future rewards by the prior probability yet.
        target_seq_idx = seq_idx;

        scalar_t reward = current_rewards.at(flock_idx, target_seq_idx, transition_idx+1, provider_idx, 0) *
                    influence_model.at(flock_idx, seq_idx, transition_idx, cluster_idx);
        scalar_t punishment = current_rewards.at(flock_idx, target_seq_idx, transition_idx+1, provider_idx, 1) *
                     influence_model.at(flock_idx, seq_idx, transition_idx, cluster_idx);

        // The reward for transitioning to this cluster is the probability of getting there when transitioning,
        // times the undiscounted reward.
        cluster_rewards.at(flock_idx, seq_idx, provider_idx, 0, cluster_idx) = reward;
        cluster_rewards.at(flock_idx, seq_idx, provider_idx, 1, cluster_idx) = punishment;
      }
    }
  }
}


void discount_rewards_iterative(
    at::Tensor frequent_sequences, // input [flock_size, n_frequent_seqs, seq_len]
    at::Tensor seq_probs_priors_clusters_context, // input [flock_size, n_frequent_seqs]
    at::Tensor current_rewards,  // input [flock_size, n_frequent_seqs, seq_lookahead, n_providers, 2]
    at::Tensor influence_model, // input [flock_size, n_frequent_seqs, seq_lookahead, n_cluster_centers]
    at::Tensor cluster_rewards, // output [flock_size, n_frequent_seq, n_providers, 2, n_cluster_centers]
    int flock_size,
    int n_frequent_seqs,
    int n_cluster_centers,
    int transition_idx,
    int n_providers)
{
    CHECK_INPUT(frequent_sequences);
    CHECK_INPUT(current_rewards);
    CHECK_INPUT(influence_model);
    CHECK_INPUT(cluster_rewards);

    const auto required_threads = flock_size * n_frequent_seqs * n_providers * n_cluster_centers;
    const int blocks = GET_BLOCK_COUNT(required_threads);
    const cudaStream_t stream = set_device_get_cuda_stream(frequent_sequences.get_device());

    AT_DISPATCH_FLOATING_TYPES(current_rewards.type(), "discount_rewards_iterative_cuda", ([&] {
        discount_rewards_iterative_cuda<scalar_t><<<blocks, max_threads_per_block, 0, stream>>>(
        TensorIndexer<int64_t, 3>(frequent_sequences), // input input[flock_size, n_frequent_seqs, seq_len]
        TensorIndexer<scalar_t, 2>(seq_probs_priors_clusters_context), // input [flock_size, n_frequent_seqs]
        TensorIndexer<scalar_t, 5>(current_rewards), // input [flock_size, n_frequent_seqs, seq_len, n_providers, 2]
        TensorIndexer<scalar_t, 4>(influence_model), // input [flock_size, n_frequent_seqs, seq_len, n_cluster_centers]
        TensorIndexer<scalar_t, 5>(cluster_rewards), // output [flock_size, n_frequent_seqs, n_providers, 2, n_cluster_centers]
        flock_size,
        n_frequent_seqs,
        n_cluster_centers,
        transition_idx,
        n_providers,
        required_threads);
    }));
}
