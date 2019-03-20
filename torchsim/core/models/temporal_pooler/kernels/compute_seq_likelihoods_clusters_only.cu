#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <kernel_helpers.h>

using namespace at;
/*
    Computes probabilites of frequently occurring sequences by checking the clusters in the history
    (does not multiply the probabilities by the prior likelihoods of the sequence occurring).
*/
template <typename scalar_t>
__global__ void compute_seq_likelihoods_clusters_only_cuda(
    const TensorIndexer<scalar_t, 3> cluster_history,  // input [flock_size, seq_lookbehind, n_cluster_centers]
    const TensorIndexer<int64_t, 3> frequent_seqs, // input [flock_size, n_frequent_seqs, seq_length]
    const TensorIndexer<scalar_t, 2> frequent_occurrences, // input [flock_size, n_frequent_seqs]
    TensorIndexer<scalar_t, 2> seq_likelihoods, // output [flock_size, n_frequent_seqs]
    int n_frequent_seqs,
    int seq_lookbehind,
    int required_threads)
{
  const int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < required_threads)
  {
    int seq_idx = id % n_frequent_seqs;
    int flock_idx = id / n_frequent_seqs;

    // The likelihood of this thread's frequent sequence being present in the history.
    scalar_t likelihood = 1.;

    // Calculate probability of the frequent sequence being in the history by
    // multiplying the probabilities of its clusters at the corresponding points in the history.
    for(int seq_position = 0; seq_position < seq_lookbehind; seq_position++)
    {
        int cluster = frequent_seqs.at(flock_idx, seq_idx, seq_position);

        if (cluster == -1)
        {
            likelihood = 0;
            break;
        }

        scalar_t cluster_prob = cluster_history.at(flock_idx, seq_position, cluster);
        likelihood *= cluster_prob;
    }


    // Store likelihood
    seq_likelihoods.at(flock_idx, seq_idx) = likelihood;
  }
}


void compute_seq_likelihoods_clusters_only(
    at::Tensor cluster_history,  // input [flock_size, seq_lookbehind, n_cluster_centers]
    at::Tensor frequent_seqs, // input [flock_size, n_frequent_seqs, seq_length]
    at::Tensor frequent_occurrences, // input [flock_size, n_frequent_seqs]
    at::Tensor seq_likelihoods, // output [flock_size, n_frequent_seqs]
    int flock_size,
    int n_frequent_seqs,
    int seq_lookbehind)
{
    CHECK_INPUT(cluster_history);
    CHECK_CUDA(frequent_seqs);
    CHECK_CUDA(frequent_occurrences);
    CHECK_INPUT(seq_likelihoods);

    const auto required_threads = flock_size * n_frequent_seqs;
    const int blocks = GET_BLOCK_COUNT(required_threads);
    const cudaStream_t stream = set_device_get_cuda_stream(cluster_history.get_device());

    AT_DISPATCH_FLOATING_TYPES(cluster_history.type(), "compute_seq_likelihoods_without_context", ([&] {
        compute_seq_likelihoods_clusters_only_cuda<scalar_t><<<blocks, max_threads_per_block, 0, stream>>>(
        TensorIndexer<scalar_t, 3>(cluster_history),
        TensorIndexer<int64_t, 3>(frequent_seqs),
        TensorIndexer<scalar_t, 2>(frequent_occurrences),
        TensorIndexer<scalar_t, 2>(seq_likelihoods),
        n_frequent_seqs,
        seq_lookbehind,
        required_threads);
    }));
}
