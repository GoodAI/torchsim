#include <ATen/ATen.h>

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <kernel_helpers.h>

#include <iostream>

using namespace at;

/*
    Computes square distances between each data point in the batch and each cluster center.
*/
template <typename scalar_t>
__global__ void compute_squared_distances_cuda(
    const TensorIndexer<scalar_t, 3> data,  // input [flock_size, batch_size, input_size]
    const TensorIndexer<scalar_t, 3> cluster_centers, // input [flock_size, n_cluster_centers, input_size]
    TensorIndexer<scalar_t, 3> distances, // output [flock_size, batch_size, n_cluster_centers]
    int n_cluster_centers,
    int input_size,
    int required_threads)
{
  // This is for some reason empirically faster than switching block.x and block.y.
  const int id = blockIdx.y * blockDim.y + threadIdx.y;

  if (id < required_threads)
  {
    const int cc_idx = id % n_cluster_centers;
    const int batch_idx = id / n_cluster_centers;
    const int flock_idx = blockIdx.x;

    scalar_t sum_compute_squared_distances = 0;
    // sum square roots of differences over input_size
    for (int input_idx = 0; input_idx < input_size; input_idx++){

        scalar_t dif = data.at(flock_idx, batch_idx, input_idx) -
                  cluster_centers.at(flock_idx, cc_idx, input_idx);
        //if(isnan(__half2float(dif))){
        if(isnan(dif)){
            dif = 0;
        }
        sum_compute_squared_distances += dif * dif;
    }
    distances.at(flock_idx, batch_idx, cc_idx) = sum_compute_squared_distances;
  }
}


void compute_squared_distances(
    at::Tensor data,  // input [flock_size, batch_size, n_cluster_centers, input_size]
    at::Tensor cluster_centers, // input [flock_size, batch_size, n_cluster_centers, input_size]
    at::Tensor distances,  // output [flock_size, batch_size, n_cluster_centers]
    int n_cluster_centers,
    int batch_size,
    int input_size,
    int flock_size)
{
    CHECK_INPUT(data);
    //TODO (UC): The check for contiguous cluster_centers needed to be disabled because we expand them before in conv SP
    //TODO (UC): However, we should still check if they are not fragmented in the memory.
    CHECK_CUDA(cluster_centers);
    CHECK_INPUT(distances);

    const int required_threads = batch_size * n_cluster_centers;
    const dim3 grid_size = dim3(flock_size, GET_BLOCK_COUNT(required_threads), 1);
    const dim3 block_size = dim3(1, max_threads_per_block, 1);

    const cudaStream_t stream = set_device_get_cuda_stream(data.get_device());

    AT_DISPATCH_FLOATING_TYPES(data.type(), "compute_squared_distances_kernel", ([&] {
//    AT_DISPATCH_FLOATING_TYPES(data.type(), "compute_squared_distances_kernel", ([&] {
        compute_squared_distances_cuda<scalar_t><<<grid_size, block_size, 0, stream>>>(
        TensorIndexer<scalar_t, 3>(data),  // input [flock_size, batch_size, input_size]
        TensorIndexer<scalar_t, 3>(cluster_centers), // input [flock_size, n_cluster_centers, input_size]
        TensorIndexer<scalar_t, 3>(distances), // output [flock_size, batch_size, n_cluster_centers]
        n_cluster_centers,
        input_size,
        required_threads);
    }));
}
