import torch
from torchsim.core import FLOAT_NEG_INF

from torchsim.core.models.expert_params import SamplingMethod
from .process import SPProcess
from .buffer import SPFlockBuffer


class SPFlockLearning(SPProcess):
    """The learning process of spatial pooler."""

    _buffer: SPFlockBuffer

    def __init__(self,
                 indices: torch.Tensor,
                 do_subflocking: bool,
                 buffer: SPFlockBuffer,
                 cluster_centers: torch.Tensor,
                 cluster_center_targets: torch.Tensor,
                 cluster_boosting_durations: torch.Tensor,
                 boosting_targets: torch.Tensor,
                 cluster_center_deltas: torch.Tensor,
                 prev_boosted_clusters: torch.Tensor,
                 n_cluster_centers: int,
                 input_size: int,
                 batch_size: int,
                 cluster_boost_threshold: int,
                 max_boost_time: int,
                 learning_rate: float,
                 learning_period: int,
                 execution_counter: torch.Tensor,
                 device='cuda',
                 boost=True,
                 sampling_method: SamplingMethod = SamplingMethod.LAST_N):
        super().__init__(indices, do_subflocking, n_cluster_centers, input_size, device)

        self.batch_size = batch_size
        self.cluster_boost_threshold = cluster_boost_threshold
        self.max_boost_time = max_boost_time
        self.learning_rate = learning_rate
        self.max_boost_threshold = max_boost_time + self.cluster_boost_threshold
        self.learning_period = learning_period  # used in boosting
        self._boost = boost
        self._sampling_method = sampling_method

        # region Constant Tensors

        # For indexing operations on the flock/cluster tensors
        self._cluster_index_matrix = torch.arange(0, self._n_cluster_centers, device=self._device,
                                                  dtype=torch.int64).view(1, -1)
        self._cluster_index_matrix = self._cluster_index_matrix.repeat(self._flock_size, 1)
        self._expert_row_indices = torch.arange(0, self._flock_size, device=self._device, dtype=torch.int64).view(
            -1, 1)

        # endregion

        self.tmp_boosting_targets = torch.zeros((self._flock_size, self._n_cluster_centers), device=self._device,
                                                dtype=torch.int64)
        self.boost_deltas = torch.zeros((self._flock_size, self._n_cluster_centers, self._input_size),
                                        device=self._device, dtype=self._float_dtype)

        self.data_batch = torch.zeros((self._flock_size, self.batch_size, self._input_size),
                                      device=self._device, dtype=self._float_dtype)

        self.last_batch_clusters = None

        self._buffer = self._get_buffer(buffer)
        self._cluster_centers = self._read_write(cluster_centers)
        self._cluster_center_targets = self._read_write(cluster_center_targets)
        self._cluster_boosting_durations = self._read_write(cluster_boosting_durations)
        self._boosting_targets = self._read_write(boosting_targets)
        self._cluster_center_deltas = self._read_write(cluster_center_deltas)
        self._prev_boosted_clusters = self._read_write(prev_boosted_clusters)

        self._execution_counter = self._read_write(execution_counter)

        self._check_dims(self._cluster_centers, self._cluster_center_targets, self._cluster_boosting_durations,
                         self._boosting_targets, self._cluster_center_deltas, self._prev_boosted_clusters)

    def _check_dims(self, cluster_centers: torch.Tensor,
                    cluster_center_targets: torch.Tensor,
                    cluster_boosting_durations: torch.Tensor,
                    boosting_targets: torch.Tensor,
                    cluster_center_deltas: torch.Tensor,
                    prev_boosted_clusters: torch.Tensor):
        assert cluster_centers.size() == (self._flock_size, self._n_cluster_centers, self._input_size)
        assert cluster_center_targets.size() == (self._flock_size, self._n_cluster_centers, self._input_size)
        assert cluster_boosting_durations.size() == (self._flock_size, self._n_cluster_centers)
        assert boosting_targets.size() == (self._flock_size, self._n_cluster_centers)
        assert cluster_center_deltas.size() == (self._flock_size, self._n_cluster_centers, self._input_size)
        assert prev_boosted_clusters.size() == (self._flock_size, self._n_cluster_centers)

    def run(self):
        """Sample batch from buffer and train the spatial pooler on it.

        Trains the flock by sampling from the buffer, computing targets, and moving the cluster centers.
        Training happens by having compute_cluster_targets and boost_clusters set parts of the class variable
        "cluster_center_targets". This variable is then used to
        """

        cluster_batch, variance_batch = self._sample_batch(self._cluster_centers, self.data_batch, self._buffer)

        self._compute_cluster_targets(self.data_batch, cluster_batch, self._cluster_center_targets)
        if self._boost:
            self._boost_clusters(cluster_batch, variance_batch, self._cluster_boosting_durations,
                                 self._boosting_targets,
                                 self._prev_boosted_clusters, self.tmp_boosting_targets, self._expert_row_indices,
                                 self._cluster_index_matrix)
        self._compute_deltas(self._cluster_centers, self._boosting_targets, self._cluster_center_deltas,
                             self._cluster_center_targets, self._prev_boosted_clusters, self.boost_deltas)
        self._move_cluster_centers(self._cluster_centers, self._cluster_center_deltas)

        # increase the execution counter
        self._execution_counter += 1

    def _sample_batch(self, cluster_centers, data_batch, buffer):
        """Sample the batch data for learning.

        This works differently based on self._sampling_method. Currently only LAST_N and BALANCED are supported.
        The LAST_N method will sample last N items from the buffer for each expert in the process.
        The BALANCED will sample so that the probability of each item being picked up is inversely proportional to the
        number of items which belong to the same cluster center as the picked item. All cluster centers should
        therefore receive the same amount of data on average.

        Args:
            cluster_centers: The cluster centers of this process.
            data_batch: The output where the sampled inputs should go.
            buffer: The spatial pooler buffer.
        """
        if self._sampling_method == SamplingMethod.LAST_N:
            # [flock_size, batch_size, input_size]
            buffer.sample_learning_batch(self.batch_size, data_batch, self._sampling_method)

            # Recompute the closest cluster centers for this batch of data
            # [flock_size, batch_size, n_cluster_centers], [flock_size, n_cluster_centers]
            cluster_batch, variance_batch = self.compute_closest_cluster_centers(cluster_centers, data_batch)
        else:
            # Recompute the closest cluster centers for ALL items in the buffer

            # [flock_size, batch_size, n_cluster_centers], [flock_size, n_cluster_centers]
            # NOTE: 'variance_batch' here is actually variance of cluster over the whole buffer, but we want to use it.
            all_clusters, variance_batch = self.compute_closest_cluster_centers(
                cluster_centers, buffer.inputs.get_stored_data())

            buffer.clusters.set_stored_data(all_clusters)

            # [flock_size, batch_size, input_size]
            indices = buffer.sample_learning_batch(self.batch_size, data_batch, self._sampling_method)

            indices = indices.view(
                self._flock_size, self.batch_size, 1).expand(self._flock_size, self.batch_size, self._n_cluster_centers)
            cluster_batch = torch.gather(buffer.clusters.get_stored_data(), dim=1, index=indices)

        # This could be optimized by passing self.last_batch_clusters into compute_closest_cluster_centers().
        self.last_batch_clusters = cluster_batch
        
        return cluster_batch, variance_batch

    @staticmethod
    def _compute_cluster_targets(data_batch: torch.Tensor,
                                 cluster_batch: torch.Tensor,
                                 cluster_center_targets: torch.Tensor):
        """Computes the mean values of the datapoints assigned to each cluster center in each flock member.

        This function performs the following pseudo code in a vectorised fashion:

        flock_cluster_means <- []
        For each flock member:
            cluster_means <- []
            For each cluster:
                data <- input data for this flock member assigned to this cluster
                cluster_means.append(mean(data))
            flock_cluster_means.append(cluster_means)

        See test_SPFlock.test_learning_with_boosting for a concrete version of the above.

        Args:
            data_batch (torch.Tensor): [flock_size, batch_size, input_size] - A batch of inputs drawn from an
            SPFlockBuffer instance.
            cluster_batch (torch.Tensor): [flock_size, n_clusters, input_size] - A batch of one-hot cluster centers
            corresponding to the data_batch - which cluster was the closest one to each point from the batch.

        Returns:
            Side effectingly sets self.cluster_center_targets to the target means for each cluster center
            [flock_size, n_cluster_centers, input_size].
        """
        # [flock_size, batch_size, n_clusters] -> [flock_size, n_cluster_centers, batch_size]
        index_batch = cluster_batch.permute(0, 2, 1)

        # Sum of all points belonging to each cluster
        # [flock_size, n_clusters, batch_size] * [flock_size, batch_size, input_size]
        # -> [flock_size, n_clusters, input_size]
        sum_tensor = torch.bmm(index_batch, data_batch)

        # Number of point belonging to each cluster
        # -> [flock_size, n_cluster_centers]
        cluster_counts = torch.sum(index_batch, dim=2, keepdim=True)

        # Mean of all points belonging to each cluster
        # [flock_size, n_cluster_centers, input_size]
        torch.div(sum_tensor, cluster_counts, out=cluster_center_targets)

    def _boost_clusters(self,
                        cluster_batch: torch.Tensor,
                        variance_batch: torch.Tensor,
                        cluster_boosting_durations: torch.Tensor,
                        boosting_targets: torch.Tensor,
                        prev_boosted_clusters: torch.Tensor,
                        _tmp_boosting_targets: torch.Tensor,
                        _expert_row_indices: torch.Tensor,
                        _cluster_index_matrix: torch.Tensor):
        """Boosts the clusters which have no assigned datapoints.

        Args:
            _cluster_index_matrix:
            _expert_row_indices:
            _tmp_boosting_targets:
            prev_boosted_clusters:
            boosting_targets:
            cluster_boosting_durations:
            cluster_batch (torch.Tensor): A batch of inputs drawn from an SPFlockBuffer instance.
             Shape: (batch_size, flock_size, input_size).
            variance_batch (torch.Tensor): The variances for each cluster center in each expert.
             Shape: (batch_size, flock_size, n_cluster_centers)

        Returns:
            Side effectingly sets self.boosting_targets to the cluster centers of the targets for all clusters
            irrespective of if the cluster center is to be boosted or not. Clusters with datapoints should be
            boosted towards themselves.
        """
        # Find used and unused clusters
        # [flock_size, n_cluster_centers]
        datapoints_counts = torch.sum(cluster_batch, dim=1)
        clusters_with_no_datapoints = (datapoints_counts == 0).type(torch.int64)

        # Multiply the counter by the unused cluster tensor to reset counters for used clusters to zero
        cluster_boosting_durations *= clusters_with_no_datapoints
        # Add the unused cluster tensor to increment the counts for unused clusters and multiply by the learning period
        # because this is the number of steps that passed
        cluster_boosting_durations += clusters_with_no_datapoints * self.learning_period

        # Get binary vector indicating which experts should have their boosting targets regenerated
        regenerate_clusters = self._cluster_boost_condition(cluster_boosting_durations,
                                                            prev_boosted_clusters,  # this value is updated here
                                                            self.cluster_boost_threshold,
                                                            self.max_boost_threshold,
                                                            self.learning_period)
        # No need to run the following demanding computation when noone will recompute its cluster targets.
        # TODO: once the following computation is rewritten to be more efficient, it might be better avoid CPU branching
        # TODO: here, because now the CPU has to call synchronize in order to be able to decide if to return.
        if not regenerate_clusters.any():
            return

        regenerate_clusters = regenerate_clusters.nonzero()
        # All clusters which have no datapoints have variance -1, but we only want to assign
        # those which are to be boosted some clusters, so set them to -inf so that they are gathered at the end of the
        # sorted indices
        variance_batch.masked_fill_(prev_boosted_clusters, FLOAT_NEG_INF)

        # Sort the variances for each expert
        _, sorted_indices = torch.sort(variance_batch, dim=1, descending=True)

        # Get a count of how many clusters in an expert will be targeted by boosting clusters
        n_targets = self._n_cluster_centers - torch.sum(prev_boosted_clusters, dim=1, keepdim=True)

        # Work out which good cluster will be targeted by which boosting clusters
        # Index of the vector is the source cluster, value is the target cluster, both sorted by variance
        sorted_boosting_targets = (_cluster_index_matrix - n_targets) % n_targets

        # "Unsort" the list of targets so that they align with the natural ordering of the cluster centers
        # E.g. cc_indices: [3,1,0,2] -> [0,1,2,3]
        # First unsort the source clusters by the variance
        _tmp_boosting_targets.scatter_(dim=1, index=sorted_indices, src=sorted_boosting_targets)

        # TODO: (Adv indexing)
        # Then "unsort" the target clusters (the values) by variance
        _tmp_boosting_targets.copy_(sorted_indices[_expert_row_indices, _tmp_boosting_targets])
        # Set new boosting targets only for experts that are to regenerate their targets
        boosting_targets[regenerate_clusters] = _tmp_boosting_targets[regenerate_clusters]

    @staticmethod
    def _cluster_boost_condition(cluster_boosting_durations: torch.Tensor,
                                 prev_boosted_clusters: torch.Tensor,
                                 cluster_boost_threshold: int,
                                 max_boost_threshold: int,
                                 learning_period: int):
        """Calculates which flocks should recompute their boosting cluster center targets.

        They should be recomputed if either of this happens:
         1) some new cluster fulfills the conditions to be boosted
         2) some cluster stops to be boosted
         3) some cluster is boosted for `max_boost_threshold` steps

         Also updates the prev_boosted_clusters to reflect which are now boosted.

        Args:
            cluster_boosting_durations: [flock_size, n_cluster_centers] - how many steps in a row
            was each cluster center without any point.
            prev_boosted_clusters: [flock_size, n_cluster_center] - indicator whether each cluster was boosted the
            previous step.

        Returns:
            (tensor.ByteTensor): [flock_size] A binary tensor indicating the experts for which the boosting targets
            should be recomputed.
        """
        # If any of the unused clusters are above the threshold, boost them
        over_threshold = cluster_boosting_durations > cluster_boost_threshold

        # Work out which experts have clusters that have been boosted for too long - they are currently boosted and
        # the number of steps they are boosted (cluster_boosting_durations) is a multiplier of max_boost_threshold
        # (would have been in some step between previous and current learning)
        # [flock_size, n_cluster_centers]
        over_max = over_threshold * ((cluster_boosting_durations % max_boost_threshold) < learning_period)
        over_max_condition = over_max.any(dim=1)
        new_boost_condition = torch.ne(over_threshold, prev_boosted_clusters).any(dim=1)

        # Update previously boosted clusters
        prev_boosted_clusters.copy_(over_threshold)

        return (over_max_condition + new_boost_condition) > 0

    def _compute_deltas(self,
                        cluster_centers: torch.Tensor,
                        boosting_targets: torch.Tensor,
                        cluster_center_deltas: torch.Tensor,
                        cluster_center_targets: torch.Tensor,
                        prev_boosted_clusters: torch.Tensor,
                        _boost_deltas: torch.Tensor):
        """Compute the deltas - how much should the cluster centers move."""
        # Compute deltas for the non-boosted training
        torch.add(cluster_center_targets, alpha=-1., other=cluster_centers, out=cluster_center_deltas)

        # Set NaNs to 0
        cluster_center_deltas.masked_fill_(torch.isnan(cluster_center_deltas), 0)

        # Create indexes: so that we can for each cluster center extract its target for boosting.
        # It is done even for those which should not be boosted, but they are then zeroed.
        boosting_targets_indexes = boosting_targets.unsqueeze(dim=2).expand(self._flock_size,
                                                                            self._n_cluster_centers,
                                                                            self._input_size)

        # Compute deltas for the boosting
        boosting_cluster_centers = torch.gather(cluster_centers, dim=1, index=boosting_targets_indexes)
        torch.add(boosting_cluster_centers, alpha=-1., other=cluster_centers, out=_boost_deltas)

        # Set boosting deltas to zero for clusters which are not meant to be boosted
        _boost_deltas.mul_(prev_boosted_clusters.unsqueeze(2).type(self._float_dtype))
        cluster_center_deltas.add_(_boost_deltas)

    def _move_cluster_centers(self, cluster_centers: torch.Tensor, cluster_center_deltas: torch.Tensor):
        """Shifts the cluster centers towards the mean of the clustered data by some learning rate."""
        cluster_centers.add_(cluster_center_deltas * self.learning_rate)
