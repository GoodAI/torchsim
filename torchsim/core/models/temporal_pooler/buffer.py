from torchsim.core.models.flock.buffer import Buffer, BufferStorage


class TPFlockBuffer(Buffer):
    """A buffer for Temporal pooler.

    The buffer tracks input clusters (e.g. the output of spatial pooler) and input contexts, sequence probabilities
    and output projections.
    """

    clusters: BufferStorage
    seq_probs: BufferStorage
    outputs: BufferStorage
    contexts: BufferStorage
    actions: BufferStorage
    exploring: BufferStorage
    rewards_punishments: BufferStorage

    def __init__(self, creator, flock_size, buffer_size, n_cluster_centers, n_frequent_seqs, context_size, n_providers):
        super().__init__(creator, flock_size, buffer_size)

        self.outputs = self._create_storage("outputs", (flock_size, buffer_size, n_cluster_centers))
        self.seq_probs = self._create_storage("seq_probs", (flock_size, buffer_size, n_frequent_seqs))
        self.clusters = self._create_storage("clusters", (flock_size, buffer_size, n_cluster_centers))
        # Two contexts are passed from parent - where the parent is, and where the parent wants to be
        self.contexts = self._create_storage("contexts", (flock_size, buffer_size, n_providers, context_size))
        self.actions = self._create_storage("actions", (flock_size, buffer_size, n_cluster_centers))
        self.exploring = self._create_storage("exploring", (flock_size, buffer_size, 1))  # marks steps when the expert
        # was exploring, the last singleton dimension has to be here so that the buffer
        self.rewards_punishments = self._create_storage("rewards_punishments", (flock_size, buffer_size, 2))

