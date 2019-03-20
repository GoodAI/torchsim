import torch

from torchsim.core import get_float
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.spatial_pooler import SPFlockBuffer, SPFlockForward
from torchsim.core.utils.tensor_utils import same
from tests.core.models.spatial_pooler.test_process import create_execution_counter


class TestSPFlockForward:
    def test_forward_pass(self):
        # Test1: buffer is initially empty
        flock_size = 2
        n_cluster_centers = 3
        device = 'cuda'
        float_dtype = get_float(device)
        do_subflocking = True

        all_indices = torch.arange(flock_size, dtype=torch.int64, device=device)

        creator = AllocatingCreator(device)
        buffer = SPFlockBuffer(creator=creator, flock_size=2, input_size=1, n_cluster_centers=n_cluster_centers,
                               buffer_size=10)
        cluster_centers = torch.tensor([[[-1], [1], [2]], [[4], [5], [6]]], dtype=float_dtype, device=device)
        data = torch.tensor([[0.4], [1.1]], dtype=float_dtype, device=device)

        forward_clusters = torch.zeros((flock_size, n_cluster_centers), device=device, dtype=float_dtype)

        execution_counter = create_execution_counter(flock_size, device)

        process = SPFlockForward(all_indices, do_subflocking, buffer, cluster_centers, forward_clusters, data,
                                 n_cluster_centers=n_cluster_centers, input_size=1, execution_counter=execution_counter,
                                 device=device)

        process.run()
        result = process._forward_clusters
        ground_truth_result = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=float_dtype, device=device)

        assert same(ground_truth_result, result)

        # Test2: One expert gets the same input as last time
        data = torch.tensor([[0.4], [5]], dtype=float_dtype, device=device)
        process = SPFlockForward(all_indices, do_subflocking, buffer, cluster_centers, forward_clusters, data,
                                 n_cluster_centers=n_cluster_centers, input_size=1, execution_counter=execution_counter,
                                 device=device)
        process.run()
        result = process._forward_clusters
        ground_truth_result = torch.tensor([[0, 1, 0.], [0, 1, 0]], dtype=float_dtype, device=device)

        assert same(ground_truth_result, result)
