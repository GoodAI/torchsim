import numpy as np
import pytest
from typing import Tuple

import torch

from torchsim.core import get_float
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.expert_params import ExpertParams, SamplingMethod
from torchsim.core.models.spatial_pooler import SPFlock
from torchsim.core.models.spatial_pooler.process import SPProcess
from torchsim.core.utils.tensor_utils import same, id_to_one_hot
from tests.core.models.integration_test_utils import randomize_subflock, calculate_expected_results, \
    check_integration_results


def create_execution_counter(flock_size: int, device: str = 'cpu'):
    float_dtype = get_float(device)
    return torch.zeros((flock_size, 1), device=device, dtype=float_dtype)


def get_subflock_creation_testing_flock(flock_size=10,
                                        subflock_size=6,
                                        input_size=5,
                                        buffer_size=7,
                                        batch_size=5,
                                        n_cluster_centers=4,
                                        device='cpu',
                                        sampling_method=SamplingMethod.LAST_N) -> Tuple[SPFlock, torch.Tensor]:
    # Generate a flock with some data
    float_dtype = get_float(device)
    params = ExpertParams()
    params.flock_size = flock_size
    params.n_cluster_centers = n_cluster_centers

    params.spatial.input_size = input_size
    params.spatial.buffer_size = buffer_size
    params.spatial.batch_size = batch_size
    params.spatial.sampling_method = sampling_method

    flock = SPFlock(params, AllocatingCreator(device))

    flock.buffer.inputs.stored_data = torch.rand((flock_size, buffer_size, input_size), dtype=float_dtype,
                                                 device=device)
    flock.buffer.clusters.stored_data = torch.rand(
        (flock_size, buffer_size, n_cluster_centers), dtype=float_dtype,
        device=device)  # NOTE: This is not one-hot and thus not real data
    flock.buffer.current_ptr = torch.rand(flock_size, device=device).type(torch.int64)
    flock.buffer.total_data_written = torch.rand(flock_size, device=device).type(torch.int64)

    # The bookeeping values which are copied across
    flock.cluster_boosting_durations = torch.rand((flock_size, n_cluster_centers), device=device).type(torch.int64)
    flock.prev_boosted_clusters = torch.clamp(torch.round(torch.rand((flock_size, n_cluster_centers), device=device)),
                                              0, 1).type(torch.uint8)
    flock.boosting_targets = torch.rand((flock_size, n_cluster_centers), device=device).type(torch.int64)

    # Indices which are to be subflocked
    indices = torch.tensor(np.random.choice(flock_size, subflock_size, replace=False), dtype=torch.int64,
                           device=device)

    return flock, indices


def get_subflock_integration_testing_flock(params, subflock_size, device) -> Tuple[SPFlock, torch.Tensor, np.array]:
    flock = SPFlock(params, AllocatingCreator(device))
    float_dtype = get_float(device)

    flock.buffer.inputs.stored_data = torch.rand(flock.buffer.inputs.dims, dtype=float_dtype, device=device)
    # NOTE: This is not one-hot and thus not real data
    flock.buffer.clusters.stored_data = torch.rand(flock.buffer.clusters.dims, dtype=float_dtype, device=device)
    flock.buffer.current_ptr = torch.randint(0, params.spatial.buffer_size, flock.buffer.current_ptr.size(),
                                             dtype=torch.int64, device=device)
    flock.buffer.total_data_written = torch.randint(0, params.spatial.buffer_size,
                                                    flock.buffer.current_ptr.size(), dtype=torch.int64, device=device)

    flock.cluster_boosting_durations = torch.randint(0, 20, flock.cluster_boosting_durations.size(), dtype=torch.int64,
                                                     device=device)
    flock.prev_boosted_clusters = \
        torch.clamp(torch.randint(0, 2, flock.prev_boosted_clusters.size(), device=device), 0, 1).type(torch.uint8)
    flock.boosting_targets = torch.randint(0, params.flock_size, flock.boosting_targets.size(), dtype=torch.int64,
                                           device=device)
    flock.forward_clusters = torch.rand(flock.forward_clusters.size(), dtype=float_dtype, device=device)

    # Create a subflock and populate it
    indices_np = sorted(np.random.choice(params.flock_size, subflock_size, replace=False))
    indices = torch.tensor(list(map(int, indices_np)), dtype=torch.int64, device=device).unsqueeze(dim=1)

    return flock, indices, indices_np


class SPProcessStub(SPProcess):
    def run(self):
        pass

    def _check_dims(self, *args):
        pass


class TestSPProcess:
    def test_compute_squared_distances(self):
        input_size = 2
        flock_size = 2
        device = 'cuda'
        float_dtype = get_float(device)
        all_indices = torch.arange(flock_size, dtype=torch.int64, device=device)
        process = SPProcessStub(all_indices, do_subflocking=True, n_cluster_centers=3, input_size=input_size,
                                device=device)
        cluster_centers = torch.tensor([[[-1, 2], [1, 2], [2, 2]], [[4, 0], [5, 0], [6, 0]]], dtype=float_dtype,
                                       device=device)
        data = torch.tensor([[0.4, 1], [1.1, -1]], dtype=float_dtype, device=device).unsqueeze_(1)

        dist = process._compute_squared_distances(cluster_centers, data)
        expected_result = torch.tensor([[[2.9600, 1.3600, 3.5600]], [[9.4100, 16.2100, 25.0100]]], dtype=float_dtype,
                                       device=device)
        assert same(expected_result, dist, eps=1e-1)

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_to_one_hot_in_process(self, device):
        flock_size = 2
        n_cluster_centers = 3
        float_dtype = get_float(device)
        all_indices = torch.arange(flock_size, dtype=torch.int64, device=device)
        process = SPProcessStub(all_indices, do_subflocking=True, n_cluster_centers=n_cluster_centers, input_size=1,
                                device=device)
        closest_cluster_centers = torch.tensor([[0, 1], [2, 2], [1, 2]], dtype=torch.int64, device=device)

        result = id_to_one_hot(closest_cluster_centers, process._n_cluster_centers, dtype=float_dtype)
        expected_result = torch.tensor([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1]], [[0, 1, 0], [0, 0, 1]]],
                                       dtype=float_dtype, device=device)

        assert same(expected_result, result)

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_integrate_forward_subflock(self, device):
        flock_size = 10
        subflock_size = 6
        input_size = 5
        n_cluster_centers = 4
        buffer_size = 7
        float_dtype = get_float(device)

        params = ExpertParams()
        params.flock_size = flock_size
        params.n_cluster_centers = n_cluster_centers

        params.spatial.input_size = input_size
        params.spatial.buffer_size = buffer_size
        params.spatial.batch_size = buffer_size

        flock, indices, indices_np = get_subflock_integration_testing_flock(params, subflock_size, device)

        data = torch.rand((flock_size, input_size), dtype=float_dtype, device=device)

        forward = flock._create_forward_process(data, indices)

        should_update = [
            (flock.forward_clusters, forward._forward_clusters),
        ]

        should_not_update = [
            (flock.cluster_centers, forward._cluster_centers),
        ]

        randomize_subflock(should_update, should_not_update)

        expected_results = calculate_expected_results(should_update, should_not_update, flock_size, indices_np)

        forward.integrate()

        check_integration_results(expected_results, should_update, should_not_update)

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_integrate_learning_subflock(self, device):
        flock_size = 10
        subflock_size = 6
        input_size = 5
        n_cluster_centers = 4
        buffer_size = 7

        params = ExpertParams()
        params.flock_size = flock_size
        params.n_cluster_centers = n_cluster_centers

        params.spatial.input_size = input_size
        params.spatial.buffer_size = buffer_size
        params.spatial.batch_size = buffer_size

        flock, indices, indices_np = get_subflock_integration_testing_flock(params, subflock_size, device)

        learning = flock._create_learning_process(indices)

        should_update = [
            (flock.cluster_boosting_durations, learning._cluster_boosting_durations),
            (flock.prev_boosted_clusters, learning._prev_boosted_clusters),
            (flock.boosting_targets, learning._boosting_targets),
            (flock.cluster_centers, learning._cluster_centers),
            (flock.cluster_center_deltas, learning._cluster_center_deltas)
        ]

        should_not_update = [
            (flock.buffer.inputs.stored_data, learning._buffer.inputs.stored_data),
            (flock.buffer.clusters.stored_data, learning._buffer.clusters.stored_data),
            (flock.buffer.current_ptr, learning._buffer.current_ptr),
            (flock.buffer.total_data_written, learning._buffer.total_data_written)
        ]

        randomize_subflock(should_update, should_not_update)

        expected_results = calculate_expected_results(should_update, should_not_update, flock_size, indices_np)

        # Run the integration
        learning.integrate()

        check_integration_results(expected_results, should_update, should_not_update)
