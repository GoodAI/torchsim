from itertools import chain
from typing import List

import torch
from torchsim.core import get_float
from torchsim.core.memory.on_device import OnDevice
from torchsim.core.utils.tensor_utils import normalize_probs_, safe_id_to_one_hot


class TPOutputProjection(OnDevice):
    frequent_seqs_scaled: torch.Tensor

    def __init__(self, flock_size: int, n_frequent_seqs: int, n_cluster_centers: int, seq_length: int,
                 seq_lookahead: int, device: str):
        super().__init__(device)

        self._flock_size = flock_size
        self._float_dtype = get_float(device)
        self.device = device

        self.n_frequent_seqs = n_frequent_seqs
        self.n_cluster_centers = n_cluster_centers

        self.seq_length = seq_length
        self.seq_lookahead = seq_lookahead
        self.seq_lookbehind = self.seq_length - self.seq_lookahead

        output_prob_scaling = self._generate_prob_scaling(seq_length, seq_lookahead)
        # tensor = torch.from_numpy(np.array(output_prob_scaling)).to(dtype=self._float_dtype, device=device)
        tensor = torch.tensor(output_prob_scaling, dtype=self._float_dtype, device=device)
        self._output_prob_scaling = self._expand_output_prob_scaling(
            tensor)

        self.frequent_seqs_scaled = torch.zeros((self._flock_size, self.n_frequent_seqs,
                                                 self.n_cluster_centers), dtype=self._float_dtype, device=device)

    def _expand_output_prob_scaling(self, output_prob_scaling: torch.Tensor) -> torch.Tensor:
        # t1 = output_prob_scaling .expand(self._flock_size, self.n_frequent_seqs, self.seq_length)
        # t2 = t1.view(self._flock_size, self.n_frequent_seqs, self.seq_length, 1)
        # t3 = t2.expand(self._flock_size, self.n_frequent_seqs, self.seq_length, self.n_cluster_centers)
        # return t3
        return output_prob_scaling \
            .expand(self._flock_size, self.n_frequent_seqs, self.seq_length) \
            .view(self._flock_size, self.n_frequent_seqs, self.seq_length, 1) \
            .expand(self._flock_size, self.n_frequent_seqs, self.seq_length, self.n_cluster_centers)

    @staticmethod
    def _generate_prob_scaling(seq_length: int, seq_lookahead: int) -> List[int]:
        """Generates linear probability scaling for output projection.

        The largest weight will be put on the current cluster (last cluster of lookbehind).
        Examples:
            Lookbehind  Lookahead   Scaling
            2           1           1, _2_, 1
            3           1           1, 2, _3_, 2
            3           2           2, _3_, 2, 1
        """
        flip = False
        if seq_lookahead >= seq_length / 2:
            # If the lookahead is longer than the history, generate the reverse and then flip.
            flip = True
            seq_lookahead = seq_length - seq_lookahead - 1

        # The length of the ascending side of the sequence.
        left_part = seq_length - seq_lookahead

        # Go up from 1.
        up = range(1, left_part + 1)

        # Go down the rest of the way.
        down = reversed(range(left_part - seq_lookahead, left_part))

        result = list(chain(up, down))

        if flip:
            result.reverse()

        return result

    def compute_output_projection(self,
                                  # [flock_size, n_frequent_seqs, seq_length]
                                  frequent_seqs: torch.Tensor,
                                  seq_likelihoods: torch.Tensor,
                                  outputs: torch.Tensor,
                                  ):
        """Transfer the sequence likelihoods into the cluster space with more weight on clusters near in time.

        Compute output projection for each sequence and aggregate (sum and normalize) all sequences per flock.

        Returns:
            Tensor [flock_size, n_cluster_centers]

        """

        # Convert each cluster center id to a set of one-hot vectors corresponding to the cluster in
        # the cluster center space.
        # [flock_size, n_frequent_seqs, seq_length, n_cluster_centers]
        frequent_seqs_unrolled = safe_id_to_one_hot(frequent_seqs, self.n_cluster_centers, self._float_dtype)

        # Scale the one hot vectors and sum the individual parts of a cluster probability vector.
        # [flock_size, n_frequent_seqs, n_cluster_centers]
        torch.sum(frequent_seqs_unrolled * self._output_prob_scaling, dim=2, out=self.frequent_seqs_scaled)

        seq_likelihoods_expanded = seq_likelihoods.unsqueeze(-1).expand(
            self._flock_size, self.n_frequent_seqs, self.n_cluster_centers)

        # Scale the individual sequence cluster probabilities by their likelihoods.
        # [flock_size, n_frequent_seqs, n_cluster_centers]
        self.frequent_seqs_scaled *= seq_likelihoods_expanded

        # Sum the cluster probabilities across all sequences in each flock.
        # [flock_size, n_cluster_centers]
        torch.sum(self.frequent_seqs_scaled, dim=1, out=outputs)

        # Normalize.
        normalize_probs_(outputs, 1)

    def compute_output_projection_per_sequence(self,
                                               # [flock_size, n_frequent_seqs, seq_length]
                                               frequent_seqs: torch.Tensor,
                                               outputs: torch.Tensor,
                                               ):
        """Compute output projection for multiple sequences

        Returns:
            Tensor [flock_size, n_frequent_seqs, n_cluster_centers]
        """
        # Convert each cluster center id to a set of one-hot vectors corresponding to the cluster in
        # the cluster center space.
        # [flock_size, n_frequent_seqs, seq_length, n_cluster_centers]
        frequent_seqs_unrolled = safe_id_to_one_hot(frequent_seqs, self.n_cluster_centers, self._float_dtype)

        # Scale the one hot vectors and sum the individual parts of a cluster probability vector.
        # [flock_size, n_frequent_seqs, n_cluster_centers]
        torch.sum(frequent_seqs_unrolled * self._output_prob_scaling, dim=2, out=outputs)

        # Normalize.
        normalize_probs_(outputs, 2)

    @staticmethod
    def compute_similarity(item: torch.Tensor, sequences: torch.Tensor):
        """Compute similarity between item and each sequence from sequences
        similarity = 1 - (sum_i( abs(item_i - seq_i) ) / n_values)

        Args:
            item - item to compute similarities for, dims: (flock_size, n_values)
            sequences - sequences to compute similarities for, dims: (flock_size, n_sequences, n_values)

        Returns
            similarity - formula: similarity = 1 - (sum_i( abs(item_i - seq_i) ) / n_values),
                         dims: (flock_size, n_sequences)
        """
        flock_size, n_values = item.shape
        items = item.unsqueeze(dim=1).expand((flock_size, sequences.shape[1], n_values))
        return torch.abs(sequences - items).sum(dim=2) / -n_values + 1
