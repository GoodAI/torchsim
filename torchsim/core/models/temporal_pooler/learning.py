import torch
import math

from torchsim.core import get_float
from torchsim.core.models.temporal_pooler.buffer import TPFlockBuffer
from torchsim.core.models.temporal_pooler.kernels import tp_process_kernels
from torchsim.core.models.temporal_pooler.process import TPProcess
from torchsim.core.utils.tensor_utils import multi_unsqueeze, gather_from_dim, safe_id_to_one_hot


class TPFlockLearning(TPProcess):
    """The learning process of the temporal pooler.

      This process trains the temporal pooler. A contigious sample is drawn from the buffer and split into subbatches
      whereupon the following happens:

        Finds already known sequences in the batch and updates the statistical model (how often it has seen them).
        Unknown sequences in the batch which are probable enough (p > 0.5) are added to the model as well.

        1) Counts occurrences of sequences in the current batch.
        2) Adds occurrences of already known sequences to all encountered sequences.
        3) Identifies new sequences with enough probability in the batch.
        4) Counts these new sequences (reduces in the first occurrence in the batch).
        5) Adds the newly encountered sequences to all and sorts by occurrences.

        The subbatching of the data further parallelises the process for counting occurrences of sequences at the cost
        of more memory overhead.
    """
    _all_encountered_seqs: torch.Tensor
    _all_encountered_seq_occurrences: torch.Tensor
    _all_encountered_context_occurrences: torch.Tensor

    _frequent_seqs: torch.Tensor
    _frequent_seq_occurrences: torch.Tensor
    _frequent_context_likelihoods: torch.Tensor
    _buffer: TPFlockBuffer

    total_encountered_occurrences: torch.Tensor

    def __init__(self,
                 indices: torch.Tensor,
                 do_subflocking: bool,
                 buffer: TPFlockBuffer,
                 all_encountered_seqs: torch.Tensor,
                 all_encountered_seq_occurrences: torch.Tensor,
                 all_encountered_context_occurrences: torch.Tensor,
                 all_encountered_rewards_punishments: torch.Tensor,
                 all_encountered_exploration_attempts: torch.Tensor,
                 all_encountered_exploration_results: torch.Tensor,
                 frequent_seqs: torch.Tensor,
                 frequent_seq_occurrences: torch.Tensor,
                 frequent_context_likelihoods: torch.Tensor,
                 frequent_rewards_punishments: torch.Tensor,
                 frequent_exploration_attempts: torch.Tensor,
                 frequent_exploration_results: torch.Tensor,
                 execution_counter: torch.Tensor,
                 max_encountered_seqs: int,
                 max_new_seqs: int,
                 n_frequent_seqs: int,
                 seq_length: int,
                 seq_lookahead: int,
                 seq_lookbehind: int,
                 n_cluster_centers: int,
                 batch_size: int,
                 forgetting_limit: int,
                 context_size: int,
                 context_prior: float,
                 exploration_attempts_prior: float,
                 n_subbatches: int,
                 n_providers: int,
                 device: str):
        super().__init__(indices, do_subflocking)

        self._exploration_results_prior = 1 / n_cluster_centers

        self._device = device
        self._max_encountered_seqs = max_encountered_seqs

        self._n_frequent_seqs = n_frequent_seqs
        self._seq_length = seq_length
        self._seq_lookahead = seq_lookahead
        self._seq_lookbehind = seq_lookbehind
        self._n_cluster_centers = n_cluster_centers
        self._batch_size = batch_size

        self._forgetting_limit = forgetting_limit
        self._context_size = context_size
        self._n_subbatches = n_subbatches
        self._subbatch_overlap = 0 if n_subbatches == 1 else self._seq_length - 1
        self._combined_batch_size = self._calculate_combined_batch_size()
        self._subbatch_size = math.ceil(self._combined_batch_size / self._n_subbatches) + self._subbatch_overlap
        self._max_seqs_in_subbatch = self._subbatch_size - (self._seq_length - 1)
        self._context_prior = context_prior
        self._exploration_attempts_prior = exploration_attempts_prior
        self._n_providers = n_providers

        self._max_seqs_in_batch = self._combined_batch_size - (self._seq_length - 1)

        self._max_new_seqs = min(self._max_seqs_in_batch, max_new_seqs)
        float_dtype = get_float(device)
        self._float_dtype = float_dtype

        # How many from all_encountered_sequences are actually updated (it does not make sense to update the bottom
        #  max_seqs_in_batch because they will be rewritten by the new seqs.
        self._used_encountered_seqs = self._max_encountered_seqs - self._max_new_seqs

        self.cluster_batch = torch.zeros((self._flock_size, self._batch_size, self._n_cluster_centers), device=device,
                                         dtype=float_dtype)

        self.context_batch = torch.zeros((self._flock_size, self._batch_size, self._n_providers, self._context_size),
                                         device=device, dtype=float_dtype)

        self.rewards_punishments_batch = torch.zeros((self._flock_size, self._batch_size, 2), device=device,
                                                     dtype=float_dtype)

        self.exploring_batch = torch.zeros((self._flock_size, self._batch_size, 1),
                                           device=device, dtype=float_dtype)

        self.actions_batch = torch.zeros((self._flock_size, self._batch_size, self._n_cluster_centers),
                                         device=device, dtype=float_dtype)

        self._buffer = self._get_buffer(buffer)

        self._execution_counter = self._read_write(execution_counter)

        self._factors = self._n_cluster_centers ** torch.arange(seq_length - 1, -1, step=-1, device=device,
                                                                dtype=torch.int64)

        self._specific_setup(all_encountered_seqs, all_encountered_seq_occurrences, frequent_seqs,
                             frequent_seq_occurrences, all_encountered_context_occurrences,
                             frequent_context_likelihoods,
                             all_encountered_exploration_attempts, all_encountered_exploration_results,
                             frequent_exploration_attempts, frequent_exploration_results,
                             all_encountered_rewards_punishments, frequent_rewards_punishments)

        self._check_dims()

    def _specific_setup(self, all_encountered_seqs, all_encountered_seq_occurrences, frequent_seqs,
                        frequent_seq_occurrences, all_encountered_context_occurrences, frequent_context_likelihoods,
                        all_encountered_exploration_attempts, all_encountered_exploration_results,
                        frequent_exploration_attempts, frequent_exploration_results,
                        all_encountered_rewards_punishments, frequent_rewards_punishments):
        """Perform setup specific to the type of TP learning that is being initialised.

         For the vanilla TP, the tensors need allocated using flock_size, and all_encountered and frequent tensors
         need subflocked.
         """
        self._allocate_tensors(self._flock_size)

        self._all_encountered_seqs = self._read_write(all_encountered_seqs)
        self._all_encountered_seq_occurrences = self._read_write(all_encountered_seq_occurrences)
        self._frequent_seqs = self._read_write(frequent_seqs)
        self._frequent_seq_occurrences = self._read_write(frequent_seq_occurrences)
        self._all_encountered_context_occurrences = self._read_write(all_encountered_context_occurrences)
        self._frequent_context_likelihoods = self._read_write(frequent_context_likelihoods)
        self._all_encountered_exploration_attempts = self._read_write(all_encountered_exploration_attempts)
        self._all_encountered_exploration_results = self._read_write(all_encountered_exploration_results)
        self._frequent_exploration_attempts = self._read_write(frequent_exploration_attempts)
        self._frequent_exploration_results = self._read_write(frequent_exploration_results)
        self._all_encountered_rewards_punishments = self._read_write(all_encountered_rewards_punishments)
        self._frequent_rewards_punishments = self._read_write(frequent_rewards_punishments)

    def _check_dims(self):
        assert self._all_encountered_seqs.size() == (self._flock_size, self._max_encountered_seqs, self._seq_length)
        assert self._all_encountered_seq_occurrences.size() == (self._flock_size, self._max_encountered_seqs)
        assert self._all_encountered_context_occurrences.size() == (self._flock_size, self._max_encountered_seqs,
                                                                    self._seq_length, self._n_providers,
                                                                    self._context_size)
        assert self._frequent_seqs.size() == (self._flock_size, self._n_frequent_seqs, self._seq_length)
        assert self._frequent_seq_occurrences.size() == (self._flock_size, self._n_frequent_seqs)
        assert self._frequent_context_likelihoods.size() == (self._flock_size, self._n_frequent_seqs,
                                                             self._seq_length, self._n_providers, self._context_size)

    def _allocate_tensors(self, flock_size):

        self.cluster_subbatch = torch.zeros((flock_size, self._n_subbatches, self._subbatch_size,
                                             self._n_cluster_centers), device=self._device, dtype=self._float_dtype)

        self.context_subbatch = torch.zeros((flock_size, self._n_subbatches, self._subbatch_size,
                                             self._n_providers, self._context_size), device=self._device,
                                            dtype=self._float_dtype)

        self.rewards_punishments_subbatch = torch.zeros((flock_size, self._n_subbatches, self._subbatch_size, 2),
                                                        device=self._device, dtype=self._float_dtype)

        self.exploring_subbatch = torch.zeros((flock_size, self._n_subbatches, self._subbatch_size,
                                               1), device=self._device, dtype=self._float_dtype)

        self.actions_subbatch = torch.zeros((flock_size, self._n_subbatches, self._subbatch_size,
                                             self._n_cluster_centers), device=self._device, dtype=self._float_dtype)

        self.encountered_batch_seq_occurrences = torch.zeros((flock_size, self._max_encountered_seqs),
                                                             device=self._device, dtype=self._float_dtype)

        self.encountered_batch_context_occurrences = torch.zeros((flock_size, self._max_encountered_seqs,
                                                                  self._seq_length, self._n_providers,
                                                                  self._context_size),
                                                                 device=self._device, dtype=self._float_dtype)

        self.encountered_batch_rewards_punishments = torch.zeros((flock_size, self._max_encountered_seqs,
                                                                  self._seq_lookahead, 2),
                                                                 device=self._device, dtype=self._float_dtype)

        self.encountered_batch_exploration_attempts = torch.zeros((flock_size, self._max_encountered_seqs,
                                                                   self._seq_lookahead), device=self._device,
                                                                  dtype=self._float_dtype)

        # number of times the action in this sequence and this transition succeeded
        self.encountered_batch_exploration_results = torch.zeros((flock_size, self._max_encountered_seqs,
                                                                  self._seq_lookahead, self._n_cluster_centers),
                                                                 device=self._device,
                                                                 dtype=self._float_dtype)

        self.newly_encountered_seqs_indicator = torch.ones((flock_size, self._max_seqs_in_batch), device=self._device,
                                                           dtype=torch.int64)
        self.newly_encountered_seqs_counts = torch.zeros((flock_size, self._max_seqs_in_batch),
                                                         device=self._device,
                                                         dtype=self._float_dtype)

        self.encountered_subbatch_seq_occurrences = torch.zeros(
            (flock_size, self._n_subbatches, self._max_encountered_seqs),
            device=self._device, dtype=self._float_dtype)

        self.encountered_subbatch_context_occurrences = torch.zeros(
            (flock_size, self._n_subbatches, self._max_encountered_seqs, self._seq_length, self._n_providers,
             self._context_size), device=self._device, dtype=self._float_dtype)

        self.encountered_subbatch_rewards_punishments = torch.zeros(
            (flock_size, self._n_subbatches, self._max_encountered_seqs, self._seq_lookahead, 2), device=self._device,
            dtype=self._float_dtype)

        self.encountered_subbatch_exploration_attempts = torch.zeros(
            (flock_size, self._n_subbatches, self._max_encountered_seqs, self._seq_lookahead), device=self._device,
            dtype=self._float_dtype)

        # number of times the action in this sequence and this transition succeeded
        self.encountered_subbatch_exploration_results = torch.zeros(
            (flock_size, self._n_subbatches, self._max_encountered_seqs, self._seq_lookahead, self._n_cluster_centers),
            device=self._device, dtype=self._float_dtype)

        # we can identify only the full seqs, not partially contained in the batch
        self.most_probable_batch_seqs = torch.zeros((flock_size, self._max_seqs_in_batch,
                                                     self._seq_length), device=self._device, dtype=torch.int64)

        # probabilities of the sequences in self.most_probable_batch_seqs
        self.most_probable_batch_seq_probs = torch.zeros((flock_size,
                                                          self._max_seqs_in_batch),
                                                         device=self._device, dtype=self._float_dtype)

        self.total_encountered_occurrences = torch.zeros((flock_size,), device=self._device, dtype=self._float_dtype)

    def run(self):
        """Runs the learning process.

        Learns from batch, then applies exponential forgetting and extract frequent sequences to be used in forward.
        """

        self._buffer.clusters.sample_contiguous_batch(self._batch_size, self.cluster_batch)
        self._buffer.contexts.sample_contiguous_batch(self._batch_size, self.context_batch)

        self._buffer.exploring.sample_contiguous_batch(self._batch_size, self.exploring_batch)
        self._buffer.actions.sample_contiguous_batch(self._batch_size, self.actions_batch)

        self._buffer.rewards_punishments.sample_contiguous_batch(self._batch_size, self.rewards_punishments_batch)

        combined_cluster_batch, combined_context_batch, combined_rewards_punishments_batch, combined_exploring_batch, \
        combined_actions_batch = self._combine_flocks(self.cluster_batch, self.context_batch,
                                                      self.rewards_punishments_batch, self.exploring_batch,
                                                      self.actions_batch)

        subbatches_capacity = self._n_subbatches * self._subbatch_size - (
                self._n_subbatches - 1) * self._subbatch_overlap
        padding_length = subbatches_capacity - self._combined_batch_size

        self._subbatch(combined_cluster_batch, out=self.cluster_subbatch, padding_length=padding_length)
        self._subbatch(combined_context_batch, out=self.context_subbatch, padding_length=padding_length)
        self._subbatch(combined_rewards_punishments_batch, out=self.rewards_punishments_subbatch,
                       padding_length=padding_length)
        self._subbatch(combined_exploring_batch, out=self.exploring_subbatch, padding_length=padding_length)
        self._subbatch(combined_actions_batch, out=self.actions_subbatch, padding_length=padding_length)

        self._learn_from_batch(combined_cluster_batch,
                               self._all_encountered_seqs,
                               self._all_encountered_seq_occurrences,
                               self._all_encountered_context_occurrences,
                               self._all_encountered_rewards_punishments,
                               self._all_encountered_exploration_attempts,
                               self._all_encountered_exploration_results,
                               self.cluster_subbatch,
                               self.context_subbatch,
                               self.rewards_punishments_subbatch,
                               self.exploring_subbatch,
                               self.actions_subbatch)

        self._forget(self._all_encountered_seq_occurrences, self._all_encountered_context_occurrences,
                     self._all_encountered_rewards_punishments, self._all_encountered_exploration_attempts,
                     self.total_encountered_occurrences)
        self._extract_frequent_seqs(self._all_encountered_seqs,
                                    self._all_encountered_seq_occurrences,
                                    self._all_encountered_context_occurrences,
                                    self._all_encountered_rewards_punishments,
                                    self._all_encountered_exploration_attempts,
                                    self._all_encountered_exploration_results,
                                    self._frequent_seqs,
                                    self._frequent_seq_occurrences,
                                    self._frequent_context_likelihoods,
                                    self._frequent_rewards_punishments,
                                    self._frequent_exploration_attempts,
                                    self._frequent_exploration_results)

        # increase the execution counter
        self._execution_counter += 1

    def _learn_from_batch(self,
                          combined_cluster_batch: torch.Tensor,
                          all_encountered_seqs: torch.Tensor,
                          all_encountered_seq_occurrences: torch.Tensor,
                          all_encountered_context_occurrences: torch.Tensor,
                          all_encountered_rewards_punishments: torch.Tensor,
                          all_encountered_exploration_attempts: torch.Tensor,
                          all_encountered_exploration_results: torch.Tensor,
                          cluster_subbatch: torch.Tensor,
                          context_subbatch: torch.Tensor,
                          rewards_punishments_subbatch: torch.Tensor,
                          exploring_subbatch: torch.Tensor,
                          actions_subbatch: torch.Tensor,
                          ):
        """Learns from the given batch.

        Finds already known sequences in the batch and updates the statistical model (how often it has seen them).
        Unknown sequences in the batch which are probable enough (p > 0.5) are added to the model as well.

        1) Counts occurrences of sequences in the current batch.
        2) Adds occurrences of already known sequences to all encountered sequences.
        3) Identifies new sequences with enough probability in the batch.
        4) Counts these new sequences (reduces in the first occurrence in the batch).
        5) Adds the newly encountered sequences to all and sorts by occurrences.
        """
        self._extract_info_known_seqs(cluster_subbatch,
                                      context_subbatch,
                                      rewards_punishments_subbatch,
                                      exploring_subbatch,
                                      actions_subbatch,
                                      all_encountered_seqs,
                                      all_encountered_seq_occurrences,
                                      self.encountered_batch_seq_occurrences,
                                      self.encountered_batch_context_occurrences,
                                      self.encountered_batch_rewards_punishments,
                                      self.encountered_batch_exploration_attempts,
                                      self.encountered_batch_exploration_results,
                                      self.newly_encountered_seqs_indicator,
                                      self.encountered_subbatch_seq_occurrences,
                                      self.encountered_subbatch_context_occurrences,
                                      self.encountered_subbatch_rewards_punishments,
                                      self.encountered_subbatch_exploration_attempts,
                                      self.encountered_subbatch_exploration_results)

        self._update_knowledge_known_seqs(all_encountered_seq_occurrences,
                                          all_encountered_context_occurrences,
                                          all_encountered_rewards_punishments,
                                          all_encountered_exploration_attempts,
                                          all_encountered_exploration_results,
                                          self.encountered_batch_seq_occurrences,
                                          self.encountered_batch_context_occurrences,
                                          self.encountered_batch_rewards_punishments,
                                          self.encountered_batch_exploration_attempts,
                                          self.encountered_batch_exploration_results)

        self._identify_new_seqs(combined_cluster_batch,
                                self.newly_encountered_seqs_indicator,
                                self.most_probable_batch_seqs,
                                self.most_probable_batch_seq_probs)

        self._extract_info_new_seqs(self.most_probable_batch_seqs,
                                    self.most_probable_batch_seq_probs,
                                    self.newly_encountered_seqs_counts)

        self._update_knowledge_new_seqs(all_encountered_seqs,
                                        all_encountered_seq_occurrences,
                                        all_encountered_context_occurrences,
                                        all_encountered_rewards_punishments,
                                        all_encountered_exploration_attempts,
                                        all_encountered_exploration_results,
                                        self.most_probable_batch_seqs,
                                        self.newly_encountered_seqs_counts)

        self._sort_all_encountered(self._buffer,
                                   all_encountered_seq_occurrences,
                                   all_encountered_seqs,
                                   all_encountered_context_occurrences,
                                   all_encountered_rewards_punishments,
                                   all_encountered_exploration_attempts,
                                   all_encountered_exploration_results)

    def _extract_info_known_seqs(self,
                                 cluster_subbatch: torch.Tensor,
                                 context_subbatch: torch.Tensor,
                                 rewards_punishments_subbatch: torch.Tensor,
                                 exploring_subbatch: torch.Tensor,
                                 actions_subbatch: torch.Tensor,
                                 all_encountered_seqs: torch.Tensor,
                                 all_encountered_seq_occurrences: torch.Tensor,
                                 encountered_batch_seq_occurrences: torch.Tensor,
                                 encountered_batch_context_occurrences: torch.Tensor,
                                 encountered_batch_rewards_punishments: torch.Tensor,
                                 encountered_batch_exploration_attempts: torch.Tensor,
                                 encountered_batch_exploration_results: torch.Tensor,
                                 newly_encountered_seqs_indicator: torch.Tensor,
                                 encountered_subbatch_seq_occurrences: torch.Tensor,
                                 encountered_subbatch_context_occurrences: torch.Tensor,
                                 encountered_subbatch_rewards_punishments: torch.Tensor,
                                 encountered_subbatch_exploration_attempts: torch.Tensor,
                                 encountered_subbatch_exploration_results: torch.Tensor
                                 ):
        """For each sequence from all encountered sequences, count how much was the sequence present in the batch.

        This also removes the marks at the starts of new sequences (all are marked in the beginning). Those that still
        have the mark after this is finished are then considered to be new sequences which should be considered
        for addition into the statistical model.
        """

        # Counts known seq occurrences (for each sequence from all encountered
        # Remove marks at the starts of new sequences (if the sequence is present in all encountered and
        # have probability > 0.5) next to their first clusters
        # ( Replaces 1's in newly_encountered_seqs with 0's if the seq starting
        # at that batch position is in all_encountered_seqs)

        tp_process_kernels.count_batch_seq_occurrences(cluster_subbatch,
                                                       context_subbatch,
                                                       rewards_punishments_subbatch,
                                                       all_encountered_seqs,
                                                       all_encountered_seq_occurrences,
                                                       encountered_subbatch_seq_occurrences,
                                                       encountered_subbatch_context_occurrences,
                                                       encountered_subbatch_rewards_punishments,
                                                       newly_encountered_seqs_indicator,
                                                       exploring_subbatch,
                                                       actions_subbatch,
                                                       encountered_subbatch_exploration_attempts,
                                                       encountered_subbatch_exploration_results,
                                                       self._flock_size,
                                                       self._n_cluster_centers,
                                                       self._seq_length,
                                                       self._seq_lookbehind,
                                                       self._used_encountered_seqs,
                                                       self._max_seqs_in_subbatch,
                                                       self._context_size,
                                                       self._n_subbatches,
                                                       self._max_seqs_in_batch,
                                                       self._n_providers)

        torch.sum(encountered_subbatch_seq_occurrences, dim=1, out=encountered_batch_seq_occurrences)
        torch.sum(encountered_subbatch_context_occurrences, dim=1, out=encountered_batch_context_occurrences)
        torch.sum(encountered_subbatch_exploration_attempts, dim=1, out=encountered_batch_exploration_attempts)
        torch.sum(encountered_subbatch_exploration_results, dim=1, out=encountered_batch_exploration_results)
        torch.sum(encountered_subbatch_rewards_punishments, dim=1, out=encountered_batch_rewards_punishments)
        self._validate_newly_encountered_seqs_indicator(newly_encountered_seqs_indicator)

    @staticmethod
    def _update_knowledge_known_seqs(all_encountered_seq_occurrences: torch.Tensor,
                                     all_encountered_context_occurrences: torch.Tensor,
                                     all_encountered_rewards_punishments: torch.Tensor,
                                     all_encountered_exploration_attempts: torch.Tensor,
                                     all_encountered_exploration_results: torch.Tensor,
                                     encountered_batch_seq_occurrences: torch.Tensor,
                                     encountered_batch_context_occurrences: torch.Tensor,
                                     encountered_batch_rewards_punishments: torch.Tensor,
                                     encountered_batch_exploration_attempts: torch.Tensor,
                                     encountered_batch_exploration_results: torch.Tensor):
        """Adds known seq/context occurrences encountered in this batch to all occurrences encountered by the TP."""
        all_encountered_seq_occurrences.add_(encountered_batch_seq_occurrences)
        all_encountered_context_occurrences.add_(encountered_batch_context_occurrences)
        all_encountered_rewards_punishments.add_(encountered_batch_rewards_punishments)

        exp_attempts_expanded = all_encountered_exploration_attempts.unsqueeze(3).expand(
            all_encountered_exploration_results.size())

        new_results = exp_attempts_expanded * all_encountered_exploration_results + encountered_batch_exploration_results

        all_encountered_exploration_attempts.add_(encountered_batch_exploration_attempts)
        torch.div(input=new_results, other=exp_attempts_expanded, out=all_encountered_exploration_results)

        # replace all nans with zeros because we need to count with them in the next iteration
        all_encountered_exploration_results.masked_fill_(torch.isnan(all_encountered_exploration_results),
                                                         0)

    def _identify_new_seqs(self,
                           combined_cluster_batch: torch.Tensor,
                           newly_encountered_seqs_indicator: torch.Tensor,
                           most_probable_batch_seqs: torch.Tensor,
                           most_probable_batch_seq_probs: torch.Tensor):
        """Identifies the most probable new sequences in the batch.

        At each step in the batch, this identifies which sequence is most probable and stores the probability of
        the sequence. If the probability is less than 0.5, or if the sequence is invalid (two same clusters in a row),
        the probability is set to 0.
        """

        # Iterate over whole batch and extract the most probable sequence starting at each step
        # Then stores their probabilities, but sets them to zero if the entries where
        # the probability is <= 0.5 or newly_encountered_seqs_indicator == 0. Also sets probability to zero if the
        # sequence detected is invalid (i.e. two consecutive clusters are the same)

        # Each thread iterates over self.seq_length lines in the batch. At each line identifies the cluster
        # with highest probability and writes its id into self.most_probable_batch_seqs.
        # Then writes the probability of the sequence into self.most_probable_batch_seq_probs
        tp_process_kernels.identify_new_seqs(combined_cluster_batch,
                                             newly_encountered_seqs_indicator,
                                             most_probable_batch_seqs,
                                             most_probable_batch_seq_probs,
                                             self._flock_size,
                                             self._seq_length,
                                             self._max_seqs_in_batch,
                                             self._n_cluster_centers)

    def _extract_info_new_seqs(self,
                               most_probable_batch_seqs: torch.Tensor,
                               most_probable_batch_seq_probs: torch.Tensor,
                               newly_encountered_seqs_counts: torch.Tensor):
        """Counts how many times is each unique most probable sequence present in the batch.

        This probability is accumulated in the first occurrence of this sequence in the batch and in other occurrences
        it is set to 0.
        """
        # Gets the newly encountered sequences from this batch.
        # The number of occurrences of each sequence in this batch (sum of the probabilities) is stored in
        # newly_encountered_seqs_counts next to (i.e in the index of) the first occurrence of the sequence,
        # and all other occurrences of the sequence have zeros.

        # First, sort most probable_batch_seqs by converting the sequence into a unique integer using Godelisation,
        # sorting, and gathering using the resulting sorted indices.
        sortable_batch = self._create_ordering(most_probable_batch_seqs)
        _, indices = torch.sort(sortable_batch, dim=1)
        batch_seq_indices = indices.unsqueeze(2).expand(self._flock_size, self._max_seqs_in_batch, self._seq_length)
        most_probable_batch_seqs.copy_(torch.gather(most_probable_batch_seqs, index=batch_seq_indices, dim=1))
        most_probable_batch_seq_probs.copy_(torch.gather(most_probable_batch_seq_probs, index=indices, dim=1))

        tp_process_kernels.count_unique_seqs(most_probable_batch_seqs,
                                             most_probable_batch_seq_probs,
                                             newly_encountered_seqs_counts,
                                             self._flock_size,
                                             self._seq_length,
                                             self._max_seqs_in_batch)

        # set the clusters of sequences with 0 occurrences to -1
        self._erase_unseen_seqs(most_probable_batch_seqs, newly_encountered_seqs_counts)

    def _update_knowledge_new_seqs(self,
                                   all_encountered_seqs: torch.Tensor,
                                   all_encountered_seq_occurrences: torch.Tensor,
                                   all_encountered_context_occurrences: torch.Tensor,
                                   all_encountered_rewards_punishments: torch.Tensor,
                                   all_encountered_exploration_attempts: torch.Tensor,
                                   all_encountered_exploration_results: torch.Tensor,
                                   most_probable_batch_seqs: torch.Tensor,
                                   newly_encountered_seq_counts: torch.Tensor):
        """Most probable sequences and their occurrences are written to the end of all_encountered_sequences and
        all_encountered_seq_occurrences."""
        # most_probable_batch_seqs should now only have '-1', or be unique
        # Copy most_probable_batch_seqs and their counts to the bottom of the all_encountered tensors
        sorted_newly_encountered_seq_counts, indices = torch.sort(newly_encountered_seq_counts, dim=1, descending=True)

        # We are gathering here over two dims so to gather over one dim, offset the indices in the flock dimension by
        # the size of the batch and flatten it to one dimension
        offsets = torch.arange(0, indices.numel(),
                               step=self._max_seqs_in_batch, device=self._device).view(-1, 1).expand(indices.size())
        indices_gather = (indices + offsets).view(-1)

        # Remove the flock dimension and use gather_from_dim, then reshape the result to the original shape.
        original_shape = most_probable_batch_seqs.size()
        sorted_most_probable_batch_seqs = gather_from_dim(most_probable_batch_seqs.view(-1, self._seq_length), dim=0,
                                                          indices=indices_gather)
        sorted_most_probable_batch_seqs = sorted_most_probable_batch_seqs.view(original_shape)

        all_encountered_seqs[:, -self._max_new_seqs:, :] = sorted_most_probable_batch_seqs[:, :self._max_new_seqs]
        all_encountered_seq_occurrences[:, -self._max_new_seqs:] = sorted_newly_encountered_seq_counts[:,
                                                                   :self._max_new_seqs]

        # Write the estimate of the context (half on, half off) for each new sequence - because it is expensive to
        # compute
        # At the point where we work out which sequences are new and count them
        new_context_counts = multi_unsqueeze(sorted_newly_encountered_seq_counts[:, :self._max_new_seqs], [2, 3, 4]) / 2
        new_context_counts = new_context_counts.expand(self._flock_size, self._max_new_seqs,
                                                       self._seq_length,
                                                       self._n_providers,
                                                       self._context_size)
        all_encountered_context_occurrences[:, -self._max_new_seqs:, :, :, :] = new_context_counts

        all_encountered_exploration_attempts[:, -self._max_new_seqs:, :] = self._exploration_attempts_prior

        # Initial exploration results are a 1 at the next cluster in the sequence
        exploration_results = safe_id_to_one_hot(sorted_most_probable_batch_seqs[:, :self._max_new_seqs, self._seq_lookbehind:].contiguous(), self._n_cluster_centers)
        exploration_results = exploration_results.view(self._flock_size, self._max_new_seqs, self._seq_lookahead, self._n_cluster_centers)


        all_encountered_exploration_results[:, -self._max_new_seqs:, :, :] = exploration_results

        # Initial rewards are set at 0
        all_encountered_rewards_punishments[:, -self._max_new_seqs:, :, :] = 0

    def _sort_all_encountered(self,
                              buffer: TPFlockBuffer,
                              all_encountered_seq_occurrences: torch.Tensor,
                              all_encountered_seqs: torch.Tensor,
                              all_encountered_context_occurrences: torch.Tensor,
                              all_encountered_rewards_punishments: torch.Tensor,
                              all_encountered_exploration_attempts: torch.Tensor,
                              all_encountered_exploration_results: torch.Tensor):
        """Sorts all encountered sequences and occurrences based on the occurrences in descending order."""
        # Sort the occurrences tensors
        all_encountered_seq_occurrences_sorted, indices_sorted = torch.sort(all_encountered_seq_occurrences, dim=1,
                                                                            descending=True)
        dimensions = all_encountered_seqs.size()
        all_encountered_seqs_sorted = torch.gather(all_encountered_seqs, dim=1,
                                                   index=indices_sorted.unsqueeze(dim=2).expand(dimensions))

        # WARNING: The gathers here have to gather to a temporary value before being copied back
        # as you cannot gather/write to the same variable at the same time
        context_indices = multi_unsqueeze(indices_sorted, [2, 3, 4]).expand(all_encountered_context_occurrences.size())
        all_encountered_context_occurrences_sorted = torch.gather(all_encountered_context_occurrences, dim=1,
                                                                  index=context_indices)

        # Rewards for this flock
        reward_indices = multi_unsqueeze(indices_sorted, [2, 3]).expand(all_encountered_rewards_punishments.size())
        all_encountered_rewards_punishments_sorted = torch.gather(all_encountered_rewards_punishments, dim=1,
                                                                  index=reward_indices)

        # exploration
        exploration_indices = multi_unsqueeze(indices_sorted, [2]).expand(
            all_encountered_exploration_attempts.size())
        all_encountered_exploration_attempts_sorted = torch.gather(all_encountered_exploration_attempts, dim=1,
                                                                   index=exploration_indices)

        exploration_indices = multi_unsqueeze(indices_sorted, [2, 3]).expand(
            all_encountered_exploration_results.size())
        all_encountered_exploration_results_sorted = torch.gather(all_encountered_exploration_results,
                                                                  dim=1,
                                                                  index=exploration_indices)

        all_encountered_seq_occurrences.copy_(all_encountered_seq_occurrences_sorted)
        all_encountered_seqs.copy_(all_encountered_seqs_sorted)
        all_encountered_context_occurrences.copy_(all_encountered_context_occurrences_sorted)
        all_encountered_rewards_punishments.copy_(all_encountered_rewards_punishments_sorted)
        all_encountered_exploration_attempts.copy_(all_encountered_exploration_attempts_sorted)
        all_encountered_exploration_results.copy_(all_encountered_exploration_results_sorted)

        # we have to reorder also this tensor in buffer so that it contains correct values, because the interpretation
        #  of frequent_XXX changed
        buffer.seq_probs.reorder(indices_sorted[:, :self._n_frequent_seqs])

    def _forget(self, all_encountered_seq_occurrences: torch.Tensor, all_encountered_context_occurrences: torch.Tensor,
                all_encountered_rewards_punishments: torch.Tensor, all_encountered_exploration_attempts: torch.Tensor,
                total_encountered_occurrences: torch.Tensor):
        """Exponentially forget sequences based on the forgetting limit."""
        torch.sum(all_encountered_seq_occurrences, dim=1, out=total_encountered_occurrences)
        division_factors = total_encountered_occurrences / self._forgetting_limit
        division_factors.clamp_(min=1)

        self._broadcast_forget(all_encountered_seq_occurrences, division_factors)
        self._broadcast_forget(all_encountered_context_occurrences, division_factors)
        self._broadcast_forget(all_encountered_exploration_attempts, division_factors)
        self._broadcast_forget(all_encountered_rewards_punishments, division_factors)

        total_encountered_occurrences.clamp_(max=self._forgetting_limit)
        all_encountered_exploration_attempts.clamp_(min=self._exploration_attempts_prior)

    def _extract_frequent_seqs(self,
                               all_encountered_seqs: torch.Tensor,
                               all_encountered_seq_occurrences: torch.Tensor,
                               all_encountered_context_occurrences: torch.Tensor,
                               all_encountered_rewards_punishments: torch.Tensor,
                               all_encountered_exploration_attempts: torch.Tensor,
                               all_encountered_exploration_results: torch.Tensor,
                               frequent_seqs: torch.Tensor,
                               frequent_seq_occurrences: torch.Tensor,
                               frequent_context_likelihoods: torch.Tensor,
                               frequent_rewards_punishments: torch.Tensor,
                               frequent_exploration_attempts: torch.Tensor,
                               frequent_exploration_results: torch.Tensor):
        """Extract _n_frequent_seqs sequences from all."""
        frequent_seqs.copy_(all_encountered_seqs[:, :self._n_frequent_seqs, :])
        frequent_seq_occurrences.copy_(all_encountered_seq_occurrences[:, :self._n_frequent_seqs])
        frequent_rewards_punishments.copy_(all_encountered_rewards_punishments[:, :self._n_frequent_seqs, :, :])
        frequent_exploration_attempts.copy_(all_encountered_exploration_attempts[:, :self._n_frequent_seqs, :])
        frequent_exploration_results.copy_(all_encountered_exploration_results[:, :self._n_frequent_seqs, :, :])

        # TODO (Time-Optim): these computations might be optimized by reshuffling their order?

        frequent_context_likelihoods.copy_(all_encountered_context_occurrences[:, :self._n_frequent_seqs, :, :, :])

        freq_seq_occurrences_expanded = multi_unsqueeze(frequent_seq_occurrences, [2, 3, 4])
        freq_seq_occurrences_expanded = freq_seq_occurrences_expanded.expand(frequent_context_likelihoods.size())

        # Add some diluting balancing prior
        frequent_context_likelihoods += self._context_prior

        # convert to probabilities
        torch.div(input=frequent_context_likelihoods, other=(freq_seq_occurrences_expanded + self._context_prior * 2.0),
                  out=frequent_context_likelihoods)

    def _calculate_combined_batch_size(self):
        """Returns the size of all of the subbatches concatenated together.

        In the vanilla TP, this is just the batch_size.
        """
        return self._batch_size

    def _validate_newly_encountered_seqs_indicator(self, newly_encountered_seqs_indicator):
        """Validates the newly encountered seqs indicator.

        The vanilla TP doesnt need any extra step in falsifying newly encountered sequences.
        """
        pass

    @staticmethod
    def _erase_unseen_seqs(most_probable_batch_seqs, newly_encountered_seqs_counts):
        """Set the clusters of seqs with 0 occurrences to '-1'."""
        mask = (newly_encountered_seqs_counts == 0).unsqueeze(-1).expand(most_probable_batch_seqs.size())
        most_probable_batch_seqs.masked_fill_(mask, -1)

    @staticmethod
    def _broadcast_forget(tensor: torch.Tensor, division_factors: torch.Tensor):
        dims = [1 for k in range(len(tensor.size()) - 1)]
        division_factors_unsqueezed = multi_unsqueeze(division_factors, dims)
        tensor /= division_factors_unsqueezed

    def _create_ordering(self, tensor: torch.Tensor):
        """Combine a sequence of cluster ids into a single natural number."""
        return (tensor * self._factors).sum(-1)

    def _subbatch(self, batch, out, padding_length):
        """Create _n_subbatches batches out of a single batch from the buffer.

            Assuming a single flock for simplicity, this will split that single batch into n different batches. As the
            sequences in the batch are meant to be contiguous, this function also calculates overlap for sequences which
            start in the subbatch. For a buffer batch of 5, sequence len 2, and n of 2 this will look like:

            1
            2
            3 3
              4
              5

            So that the sequence 3 -> 4 is represented in at least one of the batches. The pertinent values for
            calculating all of this are calculated in the constructor.

        """

        if padding_length > 0:
            view_dims = [1] * batch.dim()
            expand_dims = list(batch.size())
            expand_dims[1] = padding_length
            batch_padding = torch.tensor([0], dtype=self._float_dtype, device=self._device).view(view_dims).expand(
                expand_dims)

            padded_batch = torch.cat([batch, batch_padding], dim=1)
        else:
            padded_batch = batch

        ranges = []
        offset = 0
        for k in range(self._n_subbatches):
            ranges.append(torch.arange(offset, offset + self._subbatch_size))
            offset = offset + self._subbatch_size - self._subbatch_overlap

        indices = torch.cat(ranges).to(self._device)

        result = torch.index_select(padded_batch, 1, indices).view(out.size())

        out.copy_(result)

    def _combine_flocks(self, cluster_batch, context_batch, rewards_punishments_batch, exploring_batch, actions_batch):
        """Combines multiple flock batches from the buffer into a 'single flock'.

         This implementation doesnt need to do this, so it is the identity function, see subclasses
         like ConvTP for more informative implementations.
         """
        return cluster_batch, context_batch, rewards_punishments_batch, exploring_batch, actions_batch


class ConvTPFlockLearning(TPFlockLearning):
    """The learning process of the convolutional temporal pooler.

     This process is vary similar to the standard temporal pooler learning, with the exception that after sampling,
     all the sampled batches are combined into a single batch and that tensors like all_encountered_seqs are shared
     between all experts and are updated with data from all the experts that are learning this step.

     The ConvTP enables faster acquisition of sequences and lower memory requirements.
    """

    def _specific_setup(self, all_encountered_seqs, all_encountered_seq_occurrences, frequent_seqs,
                        frequent_seq_occurrences, all_encountered_context_occurrences, frequent_context_likelihoods,
                        all_encountered_exploration_attempts, all_encountered_exploration_results,
                        frequent_exploration_attempts, frequent_exploration_results,
                        all_encountered_rewards_punishments, frequent_rewards_punishments):
        """Perform setup specific to the type of TP learning that is being initialised.

         For the ConvTP, the tensors need allocated using flock_size = 1, and the all_encountered and frequent_
         tensors are not subflocked as they have size = 1
         """
        self._batch_flock_size = self._flock_size
        self._flock_size = 1
        self._allocate_tensors(self._flock_size)

        # We don't want to subflock any of these tensors as they only have flock_size of 1
        self._all_encountered_seqs = all_encountered_seqs
        self._all_encountered_seq_occurrences = all_encountered_seq_occurrences
        self._frequent_seqs = frequent_seqs
        self._frequent_seq_occurrences = frequent_seq_occurrences
        self._all_encountered_context_occurrences = all_encountered_context_occurrences
        self._frequent_context_likelihoods = frequent_context_likelihoods
        self._all_encountered_exploration_attempts = all_encountered_exploration_attempts
        self._all_encountered_exploration_results = all_encountered_exploration_results
        self._frequent_exploration_attempts = frequent_exploration_attempts
        self._frequent_exploration_results = frequent_exploration_results
        self._all_encountered_rewards_punishments = all_encountered_rewards_punishments
        self._frequent_rewards_punishments = frequent_rewards_punishments

    def _calculate_combined_batch_size(self):
        """Returns the size of all of the subbatches concatenated together.

        In the ConvTP, this is the sizes of the sampled batches from each expert + the padding
        between the batches.
        """
        return (self._batch_size + self._subbatch_overlap) * self._flock_size

    def _validate_newly_encountered_seqs_indicator(self, newly_encountered_seqs_indicator):
        """Validates the newly encountered seqs indicator.

        All sequences are by default considered new. The padding we use between the separate batches in the ConvTP is
        all zeros, which is a valid cluster representation. These therefore not real sequences are therefore not falsified
        in the kernel and have to be done so after the fact.
        """
        newly_encountered_seqs_indicator.mul_(self._combined_valid_seqs)

    def _combine_flocks(self, cluster_batch, context_batch, rewards_punishments_batch, exploring_batch, actions_batch):
        """Combines multiple flock batches from the buffer into a 'single flock'.

          Combines batches from clusters, context, exploring and actions. A padding of invalid values of length
          sequence_len - 1 is placed between each batch before combining so that sequences starting from one expert are
          not continued into sequences from another expert.

          Also creates an indicator of which sequences are valid to be used when identifying new sequences.
         """
        combined_cluster_batch = self.pad_and_combine_tensor(cluster_batch)
        combined_context_batch = self.pad_and_combine_tensor(context_batch)
        combined_rewards_punishments_batch = self.pad_and_combine_tensor(rewards_punishments_batch)
        combined_exploring_batch = self.pad_and_combine_tensor(exploring_batch)
        combined_actions_batch = self.pad_and_combine_tensor(actions_batch)

        self._combined_valid_seqs = self.create_valid_seqs_indicator()[:, :self._max_seqs_in_batch]

        return combined_cluster_batch, combined_context_batch, combined_rewards_punishments_batch, combined_exploring_batch, combined_actions_batch

    def pad_and_combine_tensor(self, tensor):
        """Pads and combines tensors together across the first dimension.
        """
        tensor_padding = torch.zeros((self._batch_flock_size, self._subbatch_overlap) + tensor.size()[2:],
                                     dtype=self._float_dtype, device=self._device)

        combined_tensor = torch.cat([tensor, tensor_padding], dim=1)
        combined_tensor = combined_tensor.view((1, -1) + tensor.size()[2:])

        return combined_tensor

    def create_valid_seqs_indicator(self):
        """Create tensor which indicates which sequences in a combined batch which can and can't be valid sequences.

         The count_batch_sequence occurrences kernel doesn't have enough information to always know which sequences cannot
         be valid so this helper function defines a binary tensor to be multiplied with the result of the kernel and invalidate
         identified sequences that are known to be invalid.
        """
        tensor_valid = torch.ones((self._batch_flock_size, self._batch_size - self._subbatch_overlap),
                                  dtype=torch.int64, device=self._device)

        # We need to mark those sequences which have any padding in it as invalid. The number of seqs that end in
        # the padded zone, is equal to the number which start in it. And we need to take those sequences which end
        # in there into account, hence the multiplication by 2.
        tensor_padding = torch.zeros((self._batch_flock_size, self._subbatch_overlap * 2),
                                     dtype=torch.int64, device=self._device)

        combined_tensor = torch.cat([tensor_valid, tensor_padding], dim=1)
        combined_tensor = combined_tensor.view((1, -1))

        return combined_tensor

    def _check_dims(self):
        assert self._all_encountered_seqs.size() == (self._flock_size, self._max_encountered_seqs, self._seq_length)
        assert self._all_encountered_seq_occurrences.size() == (self._flock_size, self._max_encountered_seqs)
        assert self._all_encountered_context_occurrences.size() == (1, self._max_encountered_seqs,
                                                                    self._seq_length, self._n_providers,
                                                                    self._context_size)
        assert self._frequent_seqs.size() == (1, self._n_frequent_seqs, self._seq_length)
        assert self._frequent_seq_occurrences.size() == (1, self._n_frequent_seqs)
        assert self._frequent_context_likelihoods.size() == (1, self._n_frequent_seqs,
                                                             self._seq_length, self._n_providers, self._context_size)
