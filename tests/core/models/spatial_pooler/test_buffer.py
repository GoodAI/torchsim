import pytest
import torch

from torchsim.core import get_float
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.expert_params import SamplingMethod
from torchsim.core.models.spatial_pooler import SPFlockBuffer
from torchsim.utils.seed_utils import set_global_seeds


def check_values_close(values: torch.Tensor, target, tolerance_perc: float, tolerance_abs_value: float = None) -> bool:
    """Checks if all values are within tolerance around the target.

    It uses either tolerance defined by percentage or by absolute values.
    """
    if tolerance_abs_value is not None:
        return ((values >= target - tolerance_abs_value) * (values <= target + tolerance_abs_value)).all()
    else:
        return ((values >= target * (1.0 - tolerance_perc)) * (values <= target * (1.0 + tolerance_perc))).all()


class TestSPFlockBuffer:
    # statistical test to test uniform distribution
    @pytest.mark.flaky(reruns=3)
    def test_sample_learning_batch_balanced_sampling(self):
        """Extract the the clusters to which the sampled data belong. Then check that each received roughly similar amount
         of points from the buffer. """

        flock_size = 2
        buffer_size = 1000
        input_size = 5
        n_cluster_centers = 3
        batch_size = 300
        device = 'cpu'
        float_dtype = get_float(device)
        creator = AllocatingCreator(device)

        buffer = SPFlockBuffer(creator, flock_size, buffer_size, input_size, n_cluster_centers)

        set_global_seeds(None)
        buffer.inputs.stored_data.random_()
        buffer.total_data_written.fill_(9999)

        def get_cluster_center(j):
            if j % 3 == 0:
                return [1, 0, 0]
            elif j % 3 == 1:
                return [0, 1, 0]
            elif j % 3 == 2:
                return [0, 0, 1]

        cluster_centers = [[get_cluster_center(i) for i in range(buffer_size)],
                           [get_cluster_center(i + 1) for i in range(buffer_size)]]
        buffer.clusters.stored_data = torch.tensor(cluster_centers, dtype=float_dtype, device=device)

        out = torch.zeros((flock_size, batch_size, input_size), dtype=float_dtype, device=device)

        buffer.sample_learning_batch(batch_size, out, sampling_method=SamplingMethod.BALANCED)

        indices = []
        for item_idx in range(batch_size):
            sampled_item = out[:, item_idx, :].view(flock_size, 1, input_size)
            # indices in the buffer which correspond to this sampled item [expert_id, index in the buffer]
            match = (buffer.inputs.stored_data == sampled_item).all(dim=2).nonzero()
            indices.append(match[:, 1])  # pick just the index
        # indices of datapoints for each expert
        indices = torch.stack(indices, dim=1)

        sampled_clusters = []
        for expert_id in range(flock_size):
            expert_clusters = buffer.clusters.stored_data[expert_id]
            expert_indices = indices[expert_id]

            sampled_clusters.append(expert_clusters.index_select(dim=0, index=expert_indices).sum(dim=0))
        sampled_clusters = torch.stack(sampled_clusters, dim=0)

        # sampled_clusters should be roughly uniform (checking +- 15)
        assert 85 <= sampled_clusters.min()
        assert 115 >= sampled_clusters.max()

    @pytest.mark.parametrize("flock_indices", [(None), ([0, 1, 2]), ([0, 2])])
    @pytest.mark.parametrize("elements_written", [2, 10, 15])
    @pytest.mark.parametrize("method", [SamplingMethod.BALANCED, SamplingMethod.UNIFORM, SamplingMethod.LAST_N])
    def test_sample_learning_batch_combinations(self, method, flock_indices, elements_written):
        flock_size = 3
        creator = AllocatingCreator('cpu')

        f_size = len(flock_indices) if flock_indices is not None else 3

        buffer = SPFlockBuffer(creator, buffer_size=20, n_cluster_centers=3, flock_size=flock_size, input_size=5)
        if flock_indices is not None:
            buffer.set_flock_indices(creator.tensor(flock_indices, dtype=torch.int64))

        buffer.total_data_written[:] = elements_written
        buffer.clusters.stored_data[:, :elements_written] = creator.tensor([0, 1, 0])
        buffer.inputs.stored_data[:, :elements_written, :] = creator.tensor([1.3, 0.2, 0.6, 0.4, 0.1])

        # use some dummy value here to check that it is rewriting all the lines in res
        dummy_value = -2.1
        sampled_data = creator.full((f_size, elements_written, 5), fill_value=dummy_value)
        buffer.sample_learning_batch(elements_written, sampled_data, method)

        assert (sampled_data == dummy_value).any().item() == 0

    @pytest.mark.parametrize("flock_indices", [None, ([0, 1, 2]), ([1, 3])])
    @pytest.mark.parametrize("method", [SamplingMethod.BALANCED, SamplingMethod.UNIFORM])
    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_sample_learning_batch_smart_sampling(self, method, flock_indices, device):
        """Test sampling methods.

        Tests if the balanced sampling chooses the data from each cluster equally,
        and if uniform sampling uniformly according to their prevalence.
        """

        buffer_size = 3000
        sixth = buffer_size // 6
        batch_size = 1800
        flock_size = 4
        n_cluster_centers = 4
        float_dtype = get_float(device)
        creator = AllocatingCreator(device)

        f_size = len(flock_indices) if flock_indices is not None else flock_size

        buffer = SPFlockBuffer(creator, buffer_size=buffer_size, n_cluster_centers=n_cluster_centers,
                               flock_size=flock_size, input_size=5)
        if flock_indices is not None:
            buffer.set_flock_indices(creator.tensor(flock_indices, dtype=torch.int64, device=device))

        buffer.data_since_last_sample[:] = 3  # just some value to check if it was correclty updated

        data0 = creator.tensor([1, 0, 0, 0, 0], dtype=float_dtype, device=device)
        data1 = creator.tensor([0, 0.2, -15, 0, 0], dtype=float_dtype, device=device)
        data2 = creator.tensor([1, 2, 3, 4, 5], dtype=float_dtype, device=device)

        buffer.total_data_written[:] = buffer_size
        buffer.clusters.stored_data[:, :sixth] = creator.tensor([1, 0, 0, 0], dtype=float_dtype, device=device)
        buffer.clusters.stored_data[:, sixth:(sixth * 2)] = creator.tensor([0, 0, 1, 0], dtype=float_dtype,
                                                                           device=device)
        buffer.clusters.stored_data[:, (sixth * 2):] = creator.tensor([0, 0, 0, 1], dtype=float_dtype, device=device)
        buffer.inputs.stored_data[:, :sixth] = data0
        buffer.inputs.stored_data[:, sixth:(sixth * 2)] = data1
        buffer.inputs.stored_data[:, (sixth * 2):] = data2

        # use some dummy value here to check that it is rewriting all the lines in res
        dummy_value = -2.1
        sampled_data = torch.full((f_size, batch_size, 5), fill_value=dummy_value, dtype=float_dtype, device=device)
        sampled_indices = buffer.sample_learning_batch(batch_size, sampled_data, method)
        sampled_indices = sampled_indices.view(
            f_size, batch_size, 1).expand(f_size, batch_size, n_cluster_centers)
        sampled_clusters = torch.gather(buffer.clusters.get_stored_data(), dim=1, index=sampled_indices)

        # all data are taken from the buffer
        assert (sampled_data == dummy_value).any().item() == 0

        n_points_from_each_cluster = sampled_clusters.sum(dim=1)

        # Its impossible to test the exact values so we check if it falls into reasonable boundaries around the
        # expected values.
        # mean number of points from clusters 0, 1, 2 and 3 should be;
        # UNIFORM:   300, 0, 300, 1200
        # BALANCED:  it is not 600, 0, 600, 600, because there is not enough data in the buffer and it is sampled
        # without replacement

        if method == SamplingMethod.UNIFORM:
            assert check_values_close(n_points_from_each_cluster[:, 0], 300, 0.2)
            assert check_values_close(n_points_from_each_cluster[:, 1], 0, 0)
            assert check_values_close(n_points_from_each_cluster[:, 2], 300, 0.2)
            assert check_values_close(n_points_from_each_cluster[:, 3], 1200, 0.1)

        else:
            assert check_values_close(n_points_from_each_cluster[:, 1], 0, 0)

            assert check_values_close(n_points_from_each_cluster[:, 0] - n_points_from_each_cluster[:, 2], 0,
                                      tolerance_perc=0, tolerance_abs_value=100)
            assert (n_points_from_each_cluster[:, 3] < 1200).all()

        # check that all indices we sampled for are reset to zero
        assert (buffer.data_since_last_sample[flock_indices] == 0).all()
