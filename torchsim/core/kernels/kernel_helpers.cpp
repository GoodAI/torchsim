#include "kernel_helpers.h"

#include <ATen/cuda/CUDAContext.h>


cudaStream_t set_device_get_cuda_stream(int device) {
    at::cuda::set_device(device);
    return at::cuda::getCurrentCUDAStream(device).stream();
}

