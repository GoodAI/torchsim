from datetime import datetime

import pytest

import torch
from torchsim.core import get_float
from torchsim.core.models.spatial_pooler.kernels import sp_process_kernels
from torchsim.utils.seed_utils import set_global_seeds
from tests.testing_utils import measure_time


@pytest.mark.skip("Run this to profile different boxes and configs")
@pytest.mark.parametrize("size, func, iters, name", [(10, (lambda x: torch.sum(x)), 10000, "Sum"),
                                                     (100, (lambda x: torch.sum(x)), 10000, "Sum"),
                                                     (1000, (lambda x: torch.sum(x)), 10000, "Sum"),
                                                     (10, (lambda x: x.nonzero()), 10000, "nonzero"),
                                                     (100, (lambda x: x.nonzero()), 10000, "nonzero"),
                                                     #(1000, (lambda x: x.nonzero()), 10000, "nonzero"),
                                                     (10, (lambda x: x > 0.5), 10000, "cond"),
                                                     (100, (lambda x: x > 0.5), 10000, "cond"),
                                                     # (1000, (lambda x: x > 0.5), 10000, "cond"),
                                                     (1, (lambda x: x + 0.2), 10000, "+1"),
                                                     (10000, (lambda x: x + 0.2), 1, "+1")])
@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_low_intesity_lots_of_calls(device, size, func, iters, name):
    float_dtype = get_float(device)
    tensor = torch.rand(size, size, dtype=float_dtype, device=device)

    start = datetime.now()
    for k in range(iters):
        a = func(tensor)
        assert a.device.type == device
    final = datetime.now() - start

    # Note: There might be problems with precision on Windows, it captures time with just milisecond precision.
    finalus = final.seconds + final.microseconds/1000000

    print(f"Repeat of {name}, on tensor of size ({size},{size}) on {device} for {iters} iters took {finalus}s")


@pytest.mark.skip("Run this to profile the kernel.")
def test_compute_squared_distances(capsys):

    @measure_time(iterations=10)
    def measured_function():
        sp_process_kernels.compute_squared_distances(data,
                                                     cluster_centers,
                                                     distances,
                                                     n_cluster_centers,
                                                     batch_size,
                                                     input_size,
                                                     flock_size)

    input_size = 64*64*3
    flock_size = 20
    batch_size = 3000
    n_cluster_centers = 20
    device = 'cuda'
    float_dtype = get_float(device)

    cluster_centers = torch.rand((flock_size, n_cluster_centers, input_size), dtype=float_dtype,
                                 device=device)
    # cluster_centers_expanded = cluster_centers.unsqueeze(dim=1).expand(flock_size, batch_size, n_cluster_centers,
    #                                                                    input_size)

    data = torch.rand((flock_size, batch_size, input_size), dtype=float_dtype, device=device)
    # data_expanded = data.unsqueeze(dim=2).expand(flock_size, batch_size, n_cluster_centers, input_size)

    distances = torch.full((flock_size, batch_size, n_cluster_centers), fill_value=-2.3, dtype=float_dtype,
                           device=device)
    with capsys.disabled():
        measured_function()


@pytest.mark.skip("Run this to profile the function gather_from_dim.")
def test_gather_from_dim(capsys):

    @measure_time(iterations=200, function_repetitions=1000)
    def measured_function():
        torch.index_select(input_tensor, 1, indices, out=result)

    device = 'cuda'
    float_dtype = get_float(device)
    input_tensor = torch.rand((10, 10, 10),  dtype=float_dtype, device=device)
    set_global_seeds(1)
    indices = torch.rand(10, dtype=float_dtype, device=device) < 0.4
    indices = indices.nonzero().squeeze(1)

    result = torch.empty(1, dtype=float_dtype, device=device)

    with capsys.disabled():
        measured_function()






