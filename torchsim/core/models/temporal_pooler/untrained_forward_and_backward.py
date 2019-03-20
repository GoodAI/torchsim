import torch

from torchsim.core import get_float
from torchsim.core.models.temporal_pooler.buffer import TPFlockBuffer
from torchsim.core.models.temporal_pooler.process import TPProcess
from torchsim.core.utils.tensor_utils import move_probs_towards_50, normalize_probs


class TPFlockUntrainedForwardAndBackward(TPProcess):

    _buffer: TPFlockBuffer

    def __init__(self,
                 indices: torch.Tensor,
                 do_subflocking: bool,
                 buffer: TPFlockBuffer,
                 cluster_data: torch.Tensor,
                 context_data: torch.Tensor,
                 reward_data: torch.Tensor,
                 projection_outputs: torch.Tensor,
                 action_outputs: torch.Tensor,
                 n_frequent_seqs,
                 n_cluster_centers,
                 device):
        super().__init__(indices, do_subflocking)
        float_dtype = get_float(device)

        self.n_cluster_centers = n_cluster_centers

        self._buffer = self._get_buffer(buffer)

        self._cluster_data = self._read(cluster_data)
        self._context_data = self._read(context_data)
        self._reward_data = self._read(reward_data)
        self._projection_outputs = self._read_write(projection_outputs)
        self._action_outputs = self._read_write(action_outputs)

        self.dummy_explore = torch.zeros((self._flock_size, 1), dtype=float_dtype, device=device)
        self.dummy_seq_probs = torch.zeros((self._flock_size, n_frequent_seqs), dtype=float_dtype, device=device)

    def run(self):
        with self._buffer.next_step():
            self._buffer.clusters.store(normalize_probs(self._cluster_data, dim=1, add_constant=True))
            self._buffer.contexts.store(move_probs_towards_50(self._context_data[:, :, 0, :]))
            self._buffer.rewards_punishments.store(self._reward_data)

            self._buffer.seq_probs.store(self.dummy_seq_probs)
            self._buffer.exploring.store(self.dummy_explore)

            # Generate uniform actions and projections
            self._projection_outputs.fill_(1.0 / self.n_cluster_centers)
            self._action_outputs.fill_(1.0 / self.n_cluster_centers)

            self._buffer.outputs.store(self._projection_outputs)
            self._buffer.actions.store(self._action_outputs)

    def _check_dims(self, *args):
        pass
