#include <torch/extension.h>

// forward declaration for func in .cu file
void buffer_store(at::Tensor destination,
                   at::Tensor flock_indices,
                   at::Tensor buffer_ptr_indices,
                   at::Tensor src,
                   int data_size,
                   int flock_size);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.doc() = "Buffer store kernel"; // optional module docstring
  m.def("buffer_store", &buffer_store, "Scatters inputs to the destination buffers at destination[flock_indices, buffer_ptr_indices]");
}

