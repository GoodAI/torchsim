#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <kernel_helpers.h>

#include <cstdint>

using namespace at;

template <typename scalar_t>
__global__ void buffer_store_kernel(
    TensorIndexer<scalar_t, 3> destination,
    const TensorIndexer<int64_t, 1> flock_indices,
    const TensorIndexer<int64_t, 1> buffer_ptr_indices,
    const TensorIndexer<scalar_t, 2> src,
    int data_size,
    int required_threads){

    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < required_threads){

        int data_idx = id % data_size;
        int flock_idx = id / data_size;

        destination.at(flock_indices[flock_idx], buffer_ptr_indices[flock_idx], data_idx) = src.at(flock_idx, data_idx);
    }
}



void buffer_store(at::Tensor destination,
                   at::Tensor flock_indices,
                   at::Tensor buffer_ptr_indices,
                   at::Tensor src,
                   int data_size,
                   int flock_size){

    CHECK_INPUT(destination);
    CHECK_INPUT(flock_indices);
    CHECK_INPUT(buffer_ptr_indices);
    CHECK_INPUT(src);

    const auto required_threads = flock_size * data_size;
    const int blocks = GET_BLOCK_COUNT(required_threads);
    const cudaStream_t stream = set_device_get_cuda_stream(destination.get_device());


    AT_DISPATCH_FLOATING_TYPES(destination.type(), "buffer_store_kernel", ([&] {
        buffer_store_kernel<scalar_t><<<blocks, max_threads_per_block, 0, stream>>>(
        TensorIndexer<scalar_t, 3> (destination),
        TensorIndexer<int64_t, 1> (flock_indices),
        TensorIndexer<int64_t, 1> (buffer_ptr_indices),
        TensorIndexer<scalar_t, 2> (src),
        data_size,
        required_threads);
    }));
}