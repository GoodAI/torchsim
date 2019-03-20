import pytest
import torch

from torchsim.core import get_float, FLOAT_NAN
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.expert_params import SamplingMethod
from torchsim.core.models.spatial_pooler import SPFlockBuffer, SPFlockLearning
from torchsim.core.utils.tensor_utils import same
from tests.core.models.spatial_pooler.test_process import create_execution_counter, get_subflock_creation_testing_flock


class TestSPFlockLearning:
    def test_learning_with_boosting(self):
        flock_size = 7
        n_cluster_centers = 5
        input_size = 3
        batch_size = 10
        do_subflocking = True
        learning_rate = 0.1
        learning_period = 10
        device = 'cuda'
        float_dtype = get_float(device)

        all_indices = torch.arange(flock_size, dtype=torch.int64, device=device).unsqueeze(dim=1)

        buffer = SPFlockBuffer(AllocatingCreator(device), buffer_size=20, n_cluster_centers=n_cluster_centers,
                               flock_size=flock_size, input_size=input_size)
        buffer.inputs.stored_data = torch.randn(buffer.inputs.stored_data.size(), dtype=float_dtype, device=device)

        cluster_centers = torch.rand((flock_size, n_cluster_centers, input_size), dtype=float_dtype, device=device)
        cluster_center_targets = torch.zeros((flock_size, n_cluster_centers, input_size), dtype=float_dtype,
                                             device=device)
        cluster_center_deltas = torch.zeros((flock_size, n_cluster_centers, input_size), dtype=float_dtype,
                                            device=device)
        cluster_boosting_durations = torch.full((flock_size, n_cluster_centers), fill_value=995, device=device,
                                                dtype=torch.int64)
        boosting_targets = torch.zeros((flock_size, n_cluster_centers), dtype=torch.int64, device=device)
        prev_boosted_clusters = torch.zeros((flock_size, n_cluster_centers), dtype=torch.uint8, device=device)

        buffer.current_ptr = torch.full((flock_size,), fill_value=9, dtype=torch.int64, device=device)

        buffer_data = torch.zeros((flock_size, batch_size, input_size), dtype=float_dtype, device=device)

        buffer.inputs.sample_contiguous_batch(batch_size, buffer_data)
        perm_data = buffer_data

        process = SPFlockLearning(all_indices, do_subflocking, buffer, cluster_centers, cluster_center_targets,
                                  cluster_boosting_durations,
                                  boosting_targets, cluster_center_deltas, prev_boosted_clusters,
                                  n_cluster_centers=n_cluster_centers, input_size=input_size,
                                  cluster_boost_threshold=1000, max_boost_time=200, learning_rate=learning_rate,
                                  learning_period=learning_period, batch_size=batch_size,
                                  execution_counter=create_execution_counter(flock_size, device),
                                  device=device, boost=True)

        buffer_clusters, _ = process.compute_closest_cluster_centers(cluster_centers, buffer_data)
        perm_clusters = buffer_clusters

        all_sum_data = []
        # Go through each flock and compute the means of points belonging to each cluster center
        for exp_data, exp_clust in zip(perm_data, perm_clusters):
            sum_data = []
            for c in range(process._n_cluster_centers):
                indices = exp_clust.type(torch.ByteTensor)[:, c]
                individual_data_points = exp_data[indices, :]
                if sum(indices) == 0:
                    mean_data_points = torch.full((process._input_size,), fill_value=FLOAT_NAN, dtype=float_dtype,
                                                  device=device)
                else:
                    mean_data_points = torch.mean(individual_data_points, dim=0)
                sum_data.append(mean_data_points)
            all_sum_data.append(torch.stack(sum_data))

        ground_truth = torch.stack(all_sum_data)

        previous_cluster_centers = cluster_centers.clone()

        process.run()
        process.integrate()

        # cluster center targets correctly computed
        assert same(ground_truth, cluster_center_targets, eps=1e-6)

        # cluster centers correctly moved
        deltas = cluster_centers[prev_boosted_clusters] - previous_cluster_centers[prev_boosted_clusters]
        boosting_targets_indexes = boosting_targets.unsqueeze(dim=2).expand(flock_size,
                                                                            n_cluster_centers,
                                                                            input_size)
        boosting_cluster_centers = torch.gather(previous_cluster_centers, dim=1, index=boosting_targets_indexes)
        expected_deltas = boosting_cluster_centers[prev_boosted_clusters] - previous_cluster_centers[
            prev_boosted_clusters]
        expected_deltas *= learning_rate

        assert same(expected_deltas, deltas, eps=1e-2)

        # check that cluster boosting durations was increased correctly
        assert (cluster_boosting_durations[prev_boosted_clusters == 0] == 0).all()
        assert (cluster_boosting_durations[prev_boosted_clusters] == 1005).all()

    @pytest.mark.parametrize("cluster_boosting_durations, variance_batch, cluster_batch, expected_boosting_targets",
                             [([[360, 0, 0, 370], [0, 460, 380, 0]],
                               [[0.1431, 0.1633, 0.3690, -1.], [0.2796, -1., -1., 0.7248]],
                               [[[0., 0., 1., 0.],
                                 [0., 0., 1., 0.],
                                 [0., 1., 0., 0.],
                                 [1., 0., 0., 0.]],
                                [[0., 0., 0., 1.],
                                 [0., 0., 0., 1.],
                                 [0., 0., 0., 1.],
                                 [1., 0., 0., 0.]]],
                               [[0, 1, 2, 2],
                                [0, 3, 0, 3]]),
                              ([[0, 390, 200, 390], [0, 460, 380, 0]],
                               [[0.1431, -1, -1, -1.], [0.2796, -1., -1, 0.7248]],
                               [[[1., 0., 0., 0.],
                                 [1., 0., 0., 0.],
                                 [1., 0., 0., 0.],
                                 [1., 0., 0., 0.]],
                                [[0., 0., 0., 1.],
                                 [0., 0., 0., 1.],
                                 [0., 0., 0., 1.],
                                 [1., 0., 0., 0.]]],
                               [[0, 0, 2, 2],
                                [0, 3, 0, 3]])
                              ])
    def test_boost_clusters(self, cluster_boosting_durations, variance_batch, cluster_batch, expected_boosting_targets):
        """Testing if it correctly computes the boosting targets."""
        flock_size = 2
        n_cluster_centers = 4
        input_size = 3
        batch_size = 4
        do_subflocking = True
        learning_period = 10
        device = 'cpu'
        float_dtype = get_float(device)
        cluster_boost_threshold = 370

        all_indices = torch.arange(flock_size, dtype=torch.int64, device=device).unsqueeze(dim=1)

        cluster_boosting_durations = torch.tensor(cluster_boosting_durations, dtype=torch.int64, device=device)

        variance_batch = torch.tensor(variance_batch, dtype=float_dtype, device=device)
        cluster_batch = torch.tensor(cluster_batch, dtype=float_dtype, device=device)

        prev_boosted_clusters = torch.zeros_like(cluster_boosting_durations, dtype=torch.uint8, device=device)
        boosting_targets = torch.zeros((2, 4), dtype=torch.int64, device=device)

        cluster_centers = torch.rand(flock_size, n_cluster_centers, input_size, dtype=float_dtype, device=device)
        cluster_center_targets = torch.zeros((flock_size, n_cluster_centers, input_size), dtype=float_dtype,
                                             device=device)
        cluster_center_deltas = torch.zeros((flock_size, n_cluster_centers, input_size), dtype=float_dtype,
                                            device=device)

        # Not used, but required by SPFlockLearning
        buffer = SPFlockBuffer(AllocatingCreator(device), buffer_size=20, n_cluster_centers=n_cluster_centers,
                               flock_size=flock_size,
                               input_size=input_size)

        process = SPFlockLearning(all_indices, do_subflocking, buffer, cluster_centers, cluster_center_targets,
                                  cluster_boosting_durations,
                                  boosting_targets, cluster_center_deltas, prev_boosted_clusters,
                                  n_cluster_centers=n_cluster_centers, input_size=input_size,
                                  cluster_boost_threshold=cluster_boost_threshold, max_boost_time=200,
                                  learning_rate=0.01,
                                  learning_period=learning_period, batch_size=batch_size,
                                  execution_counter=create_execution_counter(flock_size, device),
                                  device=device, boost=False)

        process._boost_clusters(cluster_batch, variance_batch, cluster_boosting_durations, boosting_targets,
                                prev_boosted_clusters, process.tmp_boosting_targets, process._expert_row_indices,
                                process._cluster_index_matrix)

        expected_boosting_targets = torch.tensor(expected_boosting_targets, dtype=torch.int64, device=device)

        assert same(expected_boosting_targets, boosting_targets)

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_non_datapoint_boosting_deltas(self, device):
        """Test if boost deltas for non-boosted clusters are zero, if they have no datapoints in the batch."""

        flock_size = 2
        n_cluster_centers = 4
        input_size = 20
        batch_size = 4
        do_subflocking = True
        learning_period = 10
        float_dtype = get_float(device)

        all_indices = torch.arange(flock_size, dtype=torch.int64, device=device)

        cluster_center_targets = torch.zeros((flock_size, n_cluster_centers, input_size), dtype=float_dtype,
                                             device=device)
        cluster_boosting_durations = torch.randint(0, 2000, (2, 4), device=device).type(torch.int64)

        variance_batch = torch.tensor([[0.1431, -1.0000, -1.0000, -1.0000],
                                       [-1.0000, -1.0000, -1.0000, 0.7248]], dtype=float_dtype, device=device)
        cluster_batch = torch.tensor([[[1., 0., 0., 0.],
                                       [1., 0., 0., 0.],
                                       [1., 0., 0., 0.],
                                       [1., 0., 0., 0.]],
                                      [[0., 0., 0., 1.],
                                       [0., 0., 0., 1.],
                                       [0., 0., 0., 1.],
                                       [0., 0., 0., 1.]]], dtype=float_dtype, device=device)

        cluster_centers = torch.rand((2, 4, 20), dtype=float_dtype, device=device)
        prev_boosted_clusters = torch.zeros_like(cluster_boosting_durations, dtype=torch.uint8, device=device)
        boosting_targets = torch.zeros((2, 4), dtype=torch.int64, device=device)
        cluster_center_deltas = torch.zeros((2, 4, 20), dtype=float_dtype, device=device)

        # Not used, but required by SPFlockLearning
        buffer = SPFlockBuffer(AllocatingCreator(device), buffer_size=20, n_cluster_centers=n_cluster_centers,
                               flock_size=flock_size,
                               input_size=input_size)

        process = SPFlockLearning(all_indices, do_subflocking, buffer, cluster_centers, cluster_center_targets,
                                  cluster_boosting_durations,
                                  boosting_targets, cluster_center_deltas, prev_boosted_clusters,
                                  n_cluster_centers=n_cluster_centers, input_size=input_size,
                                  batch_size=batch_size, cluster_boost_threshold=999, max_boost_time=100,
                                  learning_rate=0.00001, learning_period=learning_period,
                                  execution_counter=create_execution_counter(flock_size, device),
                                  device=device)

        process._boost_clusters(cluster_batch, variance_batch, cluster_boosting_durations, boosting_targets,
                                prev_boosted_clusters, process.tmp_boosting_targets, process._expert_row_indices,
                                process._cluster_index_matrix)

        # So that the datapoint-based deltas are zero
        cluster_center_targets = cluster_centers

        process._compute_deltas(cluster_centers, boosting_targets, cluster_center_deltas, cluster_center_targets,
                                prev_boosted_clusters, process.boost_deltas)

        non_boosted_cc_deltas = cluster_center_deltas[prev_boosted_clusters == 0]
        assert (torch.sum(torch.abs(non_boosted_cc_deltas)).data.item()) == 0

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_cluster_boost_condition(self, device):
        cluster_boost_threshold = 100
        max_boost_threshold = 200
        learning_period = 17

        # ex1 - do not recompute because no change in boosting and below max threshold
        # ex2 - recompute because newly boosted
        # ex3 - recompute because old is not boosted
        # ex4 - recompute because over max threshold,
        # ex5 - do not recompute because too much over max threshold
        # ex6 - recompute because 3 times over threshold

        cluster_boosting_durations = torch.tensor([[0, 1, 198, 3],
                                                   [0, 1, 198, 101],
                                                   [0, 1, 100, 3],
                                                   [0, 200, 15, 3],
                                                   [0, 1, 217, 3],
                                                   [0, 616, 15, 3]], dtype=torch.int64, device=device)
        prev_boosted_clusters = torch.tensor([[0, 0, 1, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 1, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 1, 0, 0]], dtype=torch.uint8, device=device)

        should_recompute = SPFlockLearning._cluster_boost_condition(
            cluster_boosting_durations=cluster_boosting_durations,
            prev_boosted_clusters=prev_boosted_clusters,
            cluster_boost_threshold=cluster_boost_threshold,
            max_boost_threshold=max_boost_threshold,
            learning_period=learning_period)

        expected_cluster_boosting_durations = cluster_boosting_durations.clone()
        expected_prev_boosted_clusters = torch.tensor([[0, 0, 1, 0],
                                                       [0, 0, 1, 1],
                                                       [0, 0, 0, 0],
                                                       [0, 1, 0, 0],
                                                       [0, 0, 1, 0],
                                                       [0, 1, 0, 0]], dtype=torch.uint8, device=device)

        expected_should_recompute = torch.tensor([0, 1, 1, 1, 0, 1], dtype=torch.uint8, device=device)

        assert same(expected_cluster_boosting_durations, cluster_boosting_durations)
        assert same(expected_prev_boosted_clusters, prev_boosted_clusters)
        assert same(expected_should_recompute, should_recompute)

    def test_sample_batch_sampling_last_n(self):
        device = 'cuda'
        float_dtype = get_float(device)

        flock_size = 2
        buffer_size = 5
        batch_size = 4
        input_size = 4
        subflock_size = 2
        flock, indices = get_subflock_creation_testing_flock(flock_size=flock_size,
                                                             buffer_size=buffer_size,
                                                             batch_size=batch_size,
                                                             input_size=input_size,
                                                             n_cluster_centers=4,
                                                             subflock_size=subflock_size,
                                                             device=device,
                                                             sampling_method=SamplingMethod.LAST_N)

        learning = flock._create_learning_process(indices)

        flock.buffer.clusters.stored_data = torch.tensor([[[1, 0, 0, 0],
                                                           [1, 0, 0, 0],
                                                           [0, 1, 0, 0],
                                                           [0, 0, 1, 0],
                                                           [0, 0, 0, 1]],
                                                          [[0, 0, 0, 1],
                                                           [1, 0, 0, 0],
                                                           [1, 0, 0, 0],
                                                           [0, 1, 0, 0],
                                                           [0, 0, 1, 0]]], dtype=float_dtype, device=device)

        flock.buffer.inputs.stored_data.copy_(flock.buffer.clusters.stored_data)
        flock.buffer.current_ptr[0] = 3
        flock.buffer.current_ptr[1] = 4

        # Set the cluster centers to be the same as the corresponding inputs stored in the buffer.
        ccs = flock.buffer.inputs.stored_data[:, :4, :]
        flock.cluster_centers.copy_(ccs)

        data_batch = torch.zeros((subflock_size, batch_size, input_size), dtype=float_dtype, device=device)

        expected_data_batch = torch.stack([flock.buffer.inputs.stored_data[0, :4, :],
                                           flock.buffer.inputs.stored_data[1, 1:5, :]], dim=0)

        learning._sample_batch(flock.cluster_centers, data_batch, flock.buffer)

        assert same(expected_data_batch, data_batch)

    def test_sample_batch_sampling_balanced_does_not_crash(self):
        # Note that this test only checks that the sampling is able to run without crashing.
        device = 'cuda'
        float_dtype = get_float(device)

        flock_size = 2
        buffer_size = 5
        batch_size = 4
        input_size = 4
        subflock_size = 2
        flock, indices = get_subflock_creation_testing_flock(flock_size=flock_size,
                                                             buffer_size=buffer_size,
                                                             batch_size=batch_size,
                                                             input_size=input_size,
                                                             n_cluster_centers=4,
                                                             subflock_size=subflock_size,
                                                             device=device,
                                                             sampling_method=SamplingMethod.BALANCED)

        learning = flock._create_learning_process(indices)

        flock.buffer.clusters.stored_data = torch.tensor([[[1, 0, 0, 0],
                                                           [1, 0, 0, 0],
                                                           [0, 1, 0, 0],
                                                           [0, 0, 1, 0],
                                                           [0, 0, 0, 1]],
                                                          [[0, 0, 0, 1],
                                                           [1, 0, 0, 0],
                                                           [1, 0, 0, 0],
                                                           [0, 1, 0, 0],
                                                           [0, 0, 1, 0]]], dtype=float_dtype, device=device)

        flock.buffer.inputs.stored_data.copy_(flock.buffer.clusters.stored_data)
        flock.buffer.current_ptr[0] = 3
        flock.buffer.current_ptr[1] = 4

        # Set the cluster centers to be the same as the corresponding inputs stored in the buffer.
        flock.cluster_centers = torch.tensor([[[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]],
                                              [[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]]], dtype=float_dtype, device=device)

        data_batch = torch.zeros((subflock_size, batch_size, input_size), dtype=float_dtype, device=device)

        learning._sample_batch(flock.cluster_centers, data_batch, flock.buffer)

        # Check here manually if the results are looking good.
        pass
