#include <torch/extension.h>

int get_cuda_error_code();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("get_cuda_error_code", &get_cuda_error_code, "Get last CUDA error");
}
