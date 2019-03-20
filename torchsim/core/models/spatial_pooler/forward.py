import torch

from .process import SPProcess
from .buffer import SPFlockBuffer


class SPFlockForward(SPProcess):
    """The forward pass of spatial pooler.

    This is a process and exists only for a short time. Any temporary tensors are discarded with the process.
    """

    def __init__(self,
                 indices: torch.Tensor,
                 do_subflocking: bool,
                 buffer: SPFlockBuffer,
                 cluster_centers: torch.Tensor,
                 forward_clusters: torch.Tensor,
                 data: torch.Tensor,
                 n_cluster_centers: int,
                 input_size: int,
                 execution_counter: torch.Tensor,
                 device='cuda'):
        super().__init__(indices, do_subflocking, n_cluster_centers, input_size, device)

        assert data.size()[1] == self._input_size, f"Input data have incorrect dimension ({data.size()})," \
                                                   f" expected ({self._flock_size}, {self._input_size})"

        self._data = self._read(data)
        self._cluster_centers = self._read(cluster_centers)
        self._forward_clusters = self._read_write(forward_clusters)
        self._execution_counter = self._read_write(execution_counter)

        self._buffer = self._get_buffer(buffer)

        self._check_dims(self._data, self._cluster_centers, self._forward_clusters)

    def _check_dims(self, data: torch.Tensor, cluster_centers: torch.Tensor, forward_clusters: torch.Tensor):
        assert data.size() == (self._flock_size, self._input_size)
        assert cluster_centers.size() == (self._flock_size, self._n_cluster_centers, self._input_size)
        assert forward_clusters.size() == (self._flock_size, self._n_cluster_centers)

    def run(self):
        """Calculates the closest clusters for a datapoint for each member of the flock.

        Each member of the flock receives and clusters the datapoints and returns the closest cluster for each of them.
        """

        # The commented code improves the behaviour, but is slow. So it is not used at the moment.
        # Instead, a hack is used - if the data contain nans, they are not stored into the learning buffers
        #
        # Replace nans in data with values from the closest cluster and put the improved data into the buffer
        # mask = torch.isnan(self._data)
        # cluster_idxs = self._forward_clusters.nonzero()
        # replacement_data = self._cluster_centers[cluster_idxs.chunk(chunks=2, dim=1)].squeeze(1).masked_select(mask)
        # corrected_data = self._data.masked_scatter(mask, replacement_data)
        # If uncommented, store corrected_data into self._buffer.inputs instead of self._data.

        self.compute_forward_clusters()
        # store here the data to the buffers just for experts who do not have NaNs in the inputs (hack)
        # hack commented out - will be replaced by a button in GUI
        # contains_nans = torch.isnan(self._data).any().item() == 1
        # if not contains_nans:
        with self._buffer.next_step():
            # Store the data
            self._buffer.inputs.store(self._data)
            # store the results
            self._buffer.clusters.store(self._forward_clusters)
            # increase the execution counter
            self._execution_counter += 1

    def compute_forward_clusters(self):
        # unsqueeze because self._compute_closest_cluster_centers() computes over batches of points
        # [flock_size, input_size] -> [flock_size, batch_size, input_size]
        data = torch.unsqueeze(self._data, dim=1)
        closest_clusters, _ = self.compute_closest_cluster_centers(self._cluster_centers, data)
        # squeeze because self._compute_closest_cluster_centers() computes over batches of points,
        #  but we want the result just for one point
        # [flock_size, n_cluster_centers]
        self._forward_clusters.copy_(torch.squeeze(closest_clusters, dim=1))


class ConvSPStoreToBuffer(SPProcess):
    def __init__(self,
                 indices: torch.Tensor,
                 data: torch.Tensor,
                 clusters: torch.Tensor,
                 common_buffer: SPFlockBuffer,
                 n_cluster_centers: int,
                 input_size: int,
                 device='cuda'):
        super().__init__(indices=indices, do_subflocking=True, n_cluster_centers=n_cluster_centers,
                         input_size=input_size, device=device)

        self._data = self._read(data)
        self._clusters = self._read(clusters)
        self._common_buffer = common_buffer

        self._check_dims(self._data, self._clusters)

    def _check_dims(self, data: torch.Tensor, clusters: torch.Tensor):
        assert data.size() == (self._flock_size, self._input_size)
        assert clusters.size() == (self._flock_size, self._n_cluster_centers)

    def run(self):

        # In this context, flock_size is the number of inputs/outputs that we want to store
        with self._common_buffer.next_n_steps(self._flock_size):
            self._common_buffer.inputs.store_batch(self._data.unsqueeze(0))
            self._common_buffer.clusters.store_batch(self._clusters.unsqueeze(0))
