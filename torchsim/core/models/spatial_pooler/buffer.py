import torch

from torchsim.core import SMALL_CONSTANT
from torchsim.core.models.expert_params import SamplingMethod
from torchsim.core.models.flock.buffer import BufferStorage, Buffer
from torchsim.core.utils.tensor_utils import gather_from_dim


class SPFlockBuffer(Buffer):
    """Defines a circular buffer for a flock of spatial poolers.

    The buffer keeps track of the inputs and assigned clusters for each member of the flock.
    The input buffer is of size (flock_size, buffer_size, input_size), and the corresponding cluster buffer is of size
    (flock_size, buffer_size, n_cluster_centers). These buffers are aligned and the latest point is kept track of via a
    data pointer.

    Data for training the spatial pooler is sampled from the buffer. When sampling occurs, the data from the buffers
    is extracted in reverse order staring from the data pointer. So that the oldest datapoint is the
    first in the sample.
    """

    inputs: BufferStorage
    clusters: BufferStorage

    def __init__(self, creator, flock_size, buffer_size, input_size, n_cluster_centers):
        """Initialises the buffers for a particular spatial pooler flock.

        Args:
            flock_size (int): Number of spatial poolers in the flock
            buffer_size (int): Number of elements that can be stored in the buffer before rewriting occurs
            input_size (int): The size of the input datapoints
            n_cluster_centers (int): Number of clusters that each spatial pooler can represent
        """
        super().__init__(creator, flock_size, buffer_size)
        self._n_cluster_centers = n_cluster_centers

        self.inputs = self._create_storage("inputs", (flock_size, buffer_size, input_size), force_cpu=False)
        self.clusters = self._create_storage("clusters", (flock_size, buffer_size, n_cluster_centers))

    def _compute_sampling_lambdas(self, sampling_method: SamplingMethod, flock_size: int,
                                  valid_points: torch.Tensor = None):
        if sampling_method == SamplingMethod.BALANCED:
            # Sum all cluster centers into a single vector for each flock. Then gather them into an ordered weight list
            cluster_ids = torch.argmax(self.clusters.get_stored_data(), dim=2)
            sampling_weights = self.clusters.get_stored_data()
            if valid_points is not None:
                mask = (valid_points == 0).unsqueeze(dim=2).expand(sampling_weights.size())
                sampling_weights.masked_fill(mask, 0)
                # the code above is optimization of:
                # sampling_weights[valid_points.unsqueeze(dim=2).expand(sampling_weights.size()) == 0] = 0

            sampling_weights = sampling_weights.sum(1)
            weights = torch.gather(sampling_weights, dim=1, index=cluster_ids)
        else:
            weights = torch.ones(flock_size, self.buffer_size, dtype=self._float_dtype, device=self._device)

        # Exp dist == -ln(U) / lambda == -ln(U) / (1/weights)
        exp_lambda = 1 / (weights + SMALL_CONSTANT)

        if valid_points is not None:
            exp_lambda *= valid_points.type(self._float_dtype)

        return exp_lambda

    def _compute_valid_points(self, flock_size):
        """Identify which positions in buffer are valid."""
        # Some of the buffers my not be completely filled, so we cannot sample from the whole thing...
        if self.flock_indices is None:
            total_data_written = self.total_data_written
        else:
            total_data_written = gather_from_dim(self.total_data_written, self.flock_indices, dim=0)

        if (total_data_written < self.buffer_size).any():
            total_data_written = total_data_written.view(-1, 1).expand(flock_size, self.buffer_size)
            return self.batching_tensor.view(1, -1).expand(flock_size, self.buffer_size) < total_data_written
        else:
            return None

    def _gather_inputs(self, flock_size, indices, batch_size, sampled_data_batch):
        """Gather the selected data from the buffer."""
        expanded_indices = indices.view(flock_size, batch_size, 1).expand(sampled_data_batch.shape)
        torch.gather(self.inputs.get_stored_data(), dim=1, index=expanded_indices, out=sampled_data_batch)

    def sample_learning_batch(self, batch_size, sampled_data_batch, sampling_method) -> torch.Tensor:
        """Sample batch of data from the buffer using the selected sampling method.

        Warning: Works only in the case when the cluster centers are one hot vectors!
        Warning: It does sample a really balanced batch only if buffer is infinitely large or batch size is 1,
        otherwise it might not even be possible because the sampling is witout repetition - having the same cluster
         multiple times would not help us to learn.

        Returns:
             The indices of sampled elements so that other values from the buffer (like closest cluster centers)
             can be also selected.
        """
        if sampling_method == SamplingMethod.LAST_N:
            self.inputs.sample_contiguous_batch(batch_size, sampled_data_batch)
        else:
            # Sampling method is not last n
            # We only care about the experts which are currently running
            flock_size = self.flock_size if self.flock_indices is None else len(self.flock_indices)

            valid_points = self._compute_valid_points(flock_size)

            # Exp dist == -ln(U) / lambda == -ln(U) / (1/weights)
            exp_lambda = self._compute_sampling_lambdas(sampling_method, flock_size, valid_points)
            sample = torch.rand((flock_size, self.buffer_size), device=self._device, dtype=self._float_dtype)
            sample.log_()
            sample = -sample / exp_lambda
            _, indices = torch.topk(input=sample, k=batch_size, dim=1, largest=False)

            self._gather_inputs(flock_size, indices, batch_size, sampled_data_batch)

            self.reset_data_since_last_sample()

            return indices


