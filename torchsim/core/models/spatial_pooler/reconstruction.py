import torch

from torchsim.core.models.spatial_pooler.process import SPProcess


class SPReconstruction(SPProcess):
    def __init__(self,
                 indices,
                 do_subflocking,
                 cluster_centers,
                 forward_clusters,
                 predicted_clusters,
                 current_reconstructed_input,
                 predicted_reconstructed_input,
                 n_cluster_centers,
                 input_size,
                 device):

        super().__init__(indices, do_subflocking, n_cluster_centers, input_size, device)

        self._cluster_centers = self._read(cluster_centers)
        self._forward_clusters = self._read(forward_clusters)
        self._predicted_clusters = self._read(predicted_clusters)

        self._current_reconstructed_input = self._read_write(current_reconstructed_input)
        self._predicted_reconstructed_input = self._read_write(predicted_reconstructed_input)

    def _check_dims(self, cluster_centers: torch.Tensor,
                    forward_clusters: torch.Tensor,
                    predicted_clusters: torch.Tensor,
                    current_reconstructed_input: torch.Tensor,
                    predicted_reconstructed_input: torch.Tensor):

        assert cluster_centers.size() == (self._flock_size, self._n_cluster_centers, self._input_size)
        assert forward_clusters.size() == (self._flock_size, self._n_cluster_centers)
        assert predicted_clusters.size() == (self._flock_size, self._n_cluster_centers)

        assert current_reconstructed_input.size == (self._flock_size, self._input_size)
        assert predicted_reconstructed_input.size == (self._flock_size, self._input_size)

    def run(self):
        self._reconstruct(self._cluster_centers, self._forward_clusters, self._current_reconstructed_input)
        self._reconstruct(self._cluster_centers, self._predicted_clusters, self._predicted_reconstructed_input)

    def _reconstruct(self, cluster_centers: torch.Tensor, clusters: torch.Tensor, reconstructed_input: torch.Tensor):
        self.reconstruct(self._flock_size, self._n_cluster_centers, cluster_centers, clusters, reconstructed_input)

    @staticmethod
    def reconstruct(flock_size: int, n_cluster_centers: int, cluster_centers: torch.Tensor, clusters: torch.Tensor,
                    reconstructed_input: torch.Tensor):
        """Project the given cluster centers into the input space."""
        weighted_centers = clusters.view(flock_size, n_cluster_centers, 1) * cluster_centers
        weighted_centers_sum = weighted_centers.sum(dim=1)
        reconstructed_input.copy_(weighted_centers_sum)
