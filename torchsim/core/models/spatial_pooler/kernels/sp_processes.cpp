#include <torch/extension.h>

// forward declaration for func in .cu file
void compute_squared_distances(at::Tensor data,
                               at::Tensor cluster_centers,
                               at::Tensor distances,
                               int n_cluster_centers,
                               int batch_size,
                               int input_size,
                               int flock_size);




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.doc() = "Spatial pooler process kernels"; // optional module docstring
  m.def("compute_squared_distances", &compute_squared_distances, "Compute square distances between inputs and clusters");
}

