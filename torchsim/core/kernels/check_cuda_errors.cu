#include <cuda.h>
#include <cuda_runtime.h>

int get_cuda_error_code()
{
    return (int) cudaGetLastError();
}
