#include <ATen/ATen.h>
#include <ATen/Half.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <kernel_helpers.h>


using namespace at;


//template <typename scalar_t>
__device__ __forceinline__ bool same_seqs(const int64_t* first_seq,
                                               const int64_t* second_seq,
                                               int seq_length) {
  for (int i = 0; i < seq_length; i++)
    if (first_seq[i] != second_seq[i])
        return false;

  return true;
}

template <typename scalar_t>
__global__ void count_unique_seqs(
    TensorIndexer<int64_t, 3> most_probable_batch_seqs,  // input/output [flock_size, max_seqs_in_batch, seq_length]
    const TensorIndexer<scalar_t, 2> most_probable_batch_seq_probs,  // input [flock_size, max_seqs_in_batch]
    TensorIndexer<scalar_t, 2> newly_encountered_seqs_counts,  // output [flock_size, max_seqs_in_batch]
    int seq_length,
    int max_seqs_in_batch,
    int required_threads)
{
  const int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < required_threads)
  {
    int batch_seq_idx = id % max_seqs_in_batch;
    int flock_idx = id / max_seqs_in_batch;

    scalar_t seq_occurrences = scalar_t();
    int64_t* own_seq_ptr = &(most_probable_batch_seqs.at(flock_idx, batch_seq_idx, 0));
    int64_t* next_seq_ptr;

    // If this sequence has 0 probability (i.e. it is invalid) then terminate the thread
    if (most_probable_batch_seq_probs.at(flock_idx, batch_seq_idx) == 0){
      newly_encountered_seqs_counts.at(flock_idx, batch_seq_idx) = seq_occurrences;
      return;
    }

    // The last member of the list is never the same as the next 'element'
    bool same = false;

    // Check to see that the next element in sequence is different from this one
    if (batch_seq_idx < max_seqs_in_batch - 1){
        next_seq_ptr = &(most_probable_batch_seqs.at(flock_idx, batch_seq_idx + 1, 0));
        same = same_seqs(own_seq_ptr, next_seq_ptr, seq_length);
    }

    // If the next element of this sorted list is different from the current element
    // then this is the last element of that type, so search upwards to get a count of the number of elements there are
    if (!same){
      int batch_search_idx = batch_seq_idx;
      same = true;

      while(same){
        seq_occurrences += most_probable_batch_seq_probs.at(flock_idx, batch_search_idx);
        batch_search_idx -= 1;
        same = false;

        // Put in a guard for the first element of the list
        if (batch_search_idx >= 0){
          next_seq_ptr = &(most_probable_batch_seqs.at(flock_idx, batch_search_idx, 0));
          same = same_seqs(own_seq_ptr, next_seq_ptr, seq_length);
        }
      }
    }
    newly_encountered_seqs_counts.at(flock_idx, batch_seq_idx) = seq_occurrences;
  }
}


void count_unique_seqs(at::Tensor most_probable_batch_seqs,
                            at::Tensor most_probable_batch_seq_probs,
                            at::Tensor newly_encountered_seqs_counts,
                            int flock_size,
                            int seq_length,
                            int max_seqs_in_batch)
{
    CHECK_INPUT(most_probable_batch_seqs);
    CHECK_INPUT(most_probable_batch_seq_probs);
    CHECK_INPUT(newly_encountered_seqs_counts);

    const auto required_threads = flock_size * max_seqs_in_batch;
    const int blocks = GET_BLOCK_COUNT(required_threads);
    const cudaStream_t stream = set_device_get_cuda_stream(most_probable_batch_seqs.get_device());

    AT_DISPATCH_FLOATING_TYPES(most_probable_batch_seq_probs.type(), "count_unique_seqs", ([&] {
        count_unique_seqs<scalar_t><<<blocks, max_threads_per_block, 0, stream>>>(
        TensorIndexer<int64_t, 3>(most_probable_batch_seqs),
        TensorIndexer<scalar_t, 2>(most_probable_batch_seq_probs),
        TensorIndexer<scalar_t, 2>(newly_encountered_seqs_counts),
        seq_length,
        max_seqs_in_batch,
        required_threads);
  }));
}

