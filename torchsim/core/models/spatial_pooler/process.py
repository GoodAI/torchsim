from abc import ABC
from typing import Tuple

import torch

from torchsim.core import get_float
from torchsim.core.models.flock.process import Process
from torchsim.core.models.spatial_pooler.kernels import sp_process_kernels
from torchsim.core.utils.tensor_utils import id_to_one_hot


class SPProcess(Process, ABC):
    """A spatial pooler process base class."""
    _device: str
    _n_cluster_centers: int
    _input_size = int

    def __init__(self,
                 indices: torch.Tensor,
                 do_subflocking: bool,
                 n_cluster_centers: int,
                 input_size: int,
                 device='cuda'):
        super().__init__(indices, do_subflocking)

        self._device = device
        self._n_cluster_centers = n_cluster_centers
        self._input_size = input_size
        self._float_dtype = get_float(self._device)

    def compute_closest_cluster_centers(self, cluster_centers: torch.Tensor, data: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the closest cluster for a batch of datapoints for each member of the flock.

        Each member of the flock receives and clusters the datapoints and returns the closest cluster for each of them.

        Args:
            cluster_centers: [flock_size, n_cluster_centers, input_size]
            data (torch.Tensor): [flock_size, batch_size, input_size] containing datapoints
             for each member of the flock

        Returns:
            one_hot_cluster_centers (torch.Tensor): [flock_size, batch_size, n_cluster_centers] the one-hot clustering
             of each of the datapoints given their respective spatial poolers.
            datapoint_variances (torch.Tensor): [flock_size, n_cluster_centers] - variance of all points belonging
            to each cluster center or -1 if it has zero points.
        """

        # [flock_size, batch_size, n_cluster_centers]
        distances = self._compute_squared_distances(cluster_centers, data)
        # [flock_size, batch_size]
        closest = self._closest_cluster_center(distances)

        # Get one-hot representations of the assigned cluster centers
        # [flock_size, batch_size, n_cluster_centers]
        one_hot_cluster_centers = id_to_one_hot(closest, self._n_cluster_centers, dtype=self._float_dtype)

        # Set squared distance between datapoint and irrelevant (not closest) clusters centers to zero.
        # Then sum over batch.
        # [flock_size, n_cluster_centers]
        datapoint_variances = torch.sum(one_hot_cluster_centers * distances, dim=1)

        # identify cluster centers with zero points
        # [flock_size, n_cluster_centers]
        indices = torch.sum(one_hot_cluster_centers, dim=1) == 0

        # set their variance to -1
        datapoint_variances -= indices.type(self._float_dtype)

        return one_hot_cluster_centers, datapoint_variances

    def _compute_squared_distances(self, cluster_centers: torch.Tensor, data: torch.Tensor):
        """Computes the distance between a batch of inputs per flock member and all of its cluster centers.

        Args:
            cluster_centers: [flock_size, n_cluster_centers, input_size]
            data (torch.Tensor): [flock_size, batch_size, input_size] which is batch_size inputs per member of the flock

        Returns:
            The euclidean distance between all the inputs and all of the clusters for each flock member
            [flock_size, batch_size, n_cluster_centers].
        """

        batch_size = data.size(1)
        distances = torch.zeros((self._flock_size, batch_size, self._n_cluster_centers), dtype=self._float_dtype,
                                device=self._device)

        # computes (expanded_data - expanded_cluster_centers).pow_(2).sum(3), but without need to store the
        # intermediate tensor. Consider only dimensions without NaNs (dimension with a NaN adds 0 to the total
        # distance data_point-cluster.
        sp_process_kernels.compute_squared_distances(data,
                                                     cluster_centers,
                                                     distances,
                                                     self._n_cluster_centers,
                                                     batch_size,
                                                     self._input_size,
                                                     self._flock_size)
        # [flock_size, batch_size, n_cluster_centers]
        return distances

    @staticmethod
    def _closest_cluster_center(distances: torch.Tensor):
        # [flock_size, batch_size, n_cluster_centers] -> [flock_size. batch_size, 1]
        return torch.argmin(distances, dim=2)
