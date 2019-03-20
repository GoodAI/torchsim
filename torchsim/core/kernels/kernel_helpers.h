#ifndef KERNEL_HELPERS_H
#define KERNEL_HELPERS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

#include <ATen/cuda/CUDAStream.h>
#include <THC/THCGeneral.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// TODO (?): Investigate the impact of this.
const int max_threads_per_block = 256;

#define GET_BLOCK_COUNT(required_threads) (required_threads + max_threads_per_block - 1) / max_threads_per_block  // i.e. ceil


cudaStream_t set_device_get_cuda_stream(int device);

/*
** FOR DEBUGGING WITH TensorIndexer
** Comment out the asserts and uncomment the printfs
*/
template <typename T, int N>
class TensorIndexer
{
protected:
    T* const m_ptr;
    const int m_len;
    int m_strides[N];
    int m_sizes[N];

    __device__ __forceinline__ int indexer(int idx_last) const
    {
        //if(0 > idx_last || idx_last >= m_sizes[N-1])
        //    printf("Failure @ indexer: idx_last = %d, m_sizes[N-1] = %d\n", idx_last, m_sizes[N-1]);
        assert(0 <= idx_last && idx_last < m_sizes[N-1]);
        return m_strides[N-1] * idx_last;
    }

    template <typename... Args>
    __device__ __forceinline__ int indexer(int idx, Args... rest) const
    {
        const int indexed_dim = N - sizeof...(Args) - 1;
        //if(0 > idx || idx >= m_sizes[indexed_dim])
        //    printf("Failure @ indexer: idx = %d, m_sizes[N-1] = %d\n", idx, m_sizes[indexed_dim]);
        assert(0 <= idx && idx < m_sizes[indexed_dim]);
        return m_strides[indexed_dim]*idx + indexer(rest...);
    }
public:
    TensorIndexer(at::Tensor& tensor)
    : m_ptr(tensor.data<T>()), m_len(tensor.numel())
    {
        //if(tensor.dim() != N)
        //    printf("Tensor dim of %d is not %d\n", tensor.dim(), N);
        assert(tensor.dim() == N);

        // TODO (Feat): ideally make m_strides and m_sizes const and initialize them cleverly somehow
        for (int i = 0; i < N; ++i)
        {
            m_strides[i] = tensor.stride(i);
            m_sizes[i] = tensor.size(i);
        }
    }

    template <typename... Args, typename std::enable_if<(sizeof...(Args) == N)>::type* = nullptr>
    __device__ __forceinline__ const T& at(Args... rest) const
    {
        return m_ptr[indexer(rest...)];
    }

    template <typename... Args, typename std::enable_if<(sizeof...(Args) == N)>::type* = nullptr>
    __device__ __forceinline__ T& at(Args... rest)
    {
        return m_ptr[indexer(rest...)];
    }

    __device__ __forceinline__ T& operator[](int raw_idx)
    {
        //if(0 > raw_idx || raw_idx >= m_len)
        //    printf("Failure @ indexer: raw_idx = %d, m_len = %d\n", raw_idx, m_len);
        assert(0 <= raw_idx && raw_idx < m_len);
        return m_ptr[raw_idx];
    }

    __device__ __forceinline__ const T& operator[](int raw_idx) const
    {
        //if(0 > raw_idx || raw_idx >= m_len)
        //    printf("Failure @ indexer: raw_idx = %d, m_len = %d\n", raw_idx, m_len);
        assert(0 <= raw_idx && raw_idx < m_len);
        return m_ptr[raw_idx];
    }
};

#endif /* !KERNEL_HELPERS_H */
