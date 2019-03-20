#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <kernel_helpers.h>

using namespace at;

template <typename scalar_t>
__global__ void identify_new_seqs_cuda(
    const TensorIndexer<scalar_t, 3> batch,  // input [flock_size, batch_size, n_cluster_centers]
    const TensorIndexer<int64_t, 2> newly_encountered_seqs_indicator,  // input [flock_size, batch_size]
    TensorIndexer<int64_t, 3> most_probable_batch_seqs,  // output [flock_size, max_seqs_in_batch, seq_length]
    TensorIndexer<scalar_t, 2> most_probable_batch_seq_probs,  // output [flock_size, max_seqs_in_batch]
    int seq_length,
    int max_seqs_in_batch,
    int n_cluster_centers,
    int required_threads)
{
  const int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < required_threads)
  {
    int seq_start = id % max_seqs_in_batch;
    int flock_idx = id / max_seqs_in_batch;

    // How much is the most probable seq present in the batch.
    scalar_t seq_prob = 1.0;

    // Go over seq_length lines in the batch starting at seq_start.
    int last_most_probable_cluster = -1;
    for (int seq_position = 0; seq_position < seq_length; seq_position++) {
      int batch_line = seq_start + seq_position;

      // Find the most probable cluster on the line.
      int most_probable_cluster = -1;
      scalar_t most_probable_cluster_prob = 0;
      for (int cluster_idx = 0; cluster_idx < n_cluster_centers; cluster_idx++) {
        float current_cluster_prob = batch.at(flock_idx, batch_line, cluster_idx);
        if (current_cluster_prob > most_probable_cluster_prob)
        {
          most_probable_cluster_prob = current_cluster_prob;
          most_probable_cluster = cluster_idx;
        }
      }

      // If the same cluster is selected as in last step, the sequence is illegal.
      if (last_most_probable_cluster == most_probable_cluster)
      {
        seq_prob = 0;
      }
      else
      {
        // Incrementally calculate the whole sequence probability.
        seq_prob *= most_probable_cluster_prob;
        last_most_probable_cluster = most_probable_cluster;
      }

      // Store the most probable cluster.
      most_probable_batch_seqs.at(flock_idx, seq_start, seq_position) = most_probable_cluster;
    }

    // Discard the sequence if probability is < 0.5 or if the sequence is already known.
    if (seq_prob <= 0.5 || newly_encountered_seqs_indicator.at(flock_idx, seq_start) == 0)
        seq_prob = 0;

    most_probable_batch_seq_probs.at(flock_idx, seq_start) = seq_prob;
  }
}

void identify_new_seqs(at::Tensor batch,
                            at::Tensor newly_encountered_seqs_indicator,
                            at::Tensor most_probable_batch_seqs,
                            at::Tensor most_probable_batch_seq_probs,
                            int flock_size,
                            int seq_length,
                            int max_seqs_in_batch,
                            int n_cluster_centers)
{
    CHECK_INPUT(batch);
    CHECK_INPUT(newly_encountered_seqs_indicator);
    CHECK_INPUT(most_probable_batch_seqs);
    CHECK_INPUT(most_probable_batch_seq_probs);

    const auto required_threads = flock_size * max_seqs_in_batch;
    const int blocks = GET_BLOCK_COUNT(required_threads);
    const cudaStream_t stream = set_device_get_cuda_stream(batch.get_device());

    AT_DISPATCH_FLOATING_TYPES(batch.type(), "identify_new_seqs_cuda", ([&] {
        identify_new_seqs_cuda<scalar_t><<<blocks, max_threads_per_block, 0, stream>>>(
        TensorIndexer<scalar_t, 3>(batch),
        TensorIndexer<int64_t, 2>(newly_encountered_seqs_indicator),
        TensorIndexer<int64_t, 3>(most_probable_batch_seqs),
        TensorIndexer<scalar_t, 2>(most_probable_batch_seq_probs),
        seq_length,
        max_seqs_in_batch,
        n_cluster_centers,
        required_threads);
  }));
}

