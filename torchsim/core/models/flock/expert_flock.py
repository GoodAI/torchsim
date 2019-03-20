import torch

from torchsim.core import get_float
from torchsim.core.memory.on_device import OnDevice
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ExpertParams, NUMBER_OF_CONTEXT_TYPES
from torchsim.core.models.spatial_pooler import SPFlock, ConvSPFlock
from torchsim.core.models.temporal_pooler import TPFlock


class ExpertFlock(OnDevice):
    """A container class for spatial and temporal poolers."""
    sp_flock: SPFlock
    tp_flock: TPFlock
    output_context: torch.Tensor
    params: ExpertParams

    def __init__(self, params: ExpertParams, creator: TensorCreator):
        super().__init__(creator.device)
        float_dtype = get_float(self._device)

        self.params = params
        flock_size = params.flock_size
        self.n_cluster_centers = params.n_cluster_centers
        self.seq_lookbehind = params.temporal.seq_lookbehind
        # self.context_size = self.n_cluster_centers * 2
        self.n_providers = self.params.temporal.n_providers

        # Context is: <SP_output>
        #             <Rewards>
        #             <Punishments>
        #
        #             <Pred_clusters for next step>
        #             <NaNs>
        #             <NaNs>

        # With optional NaN Padding depending on the context size in the params
        self.output_context = creator.full((flock_size, 2, NUMBER_OF_CONTEXT_TYPES, self.n_cluster_centers),
                                         fill_value=float("nan"), device=self._device, dtype=float_dtype)

        self.index_tensor = creator.arange(start=0, end=flock_size, device=self._device).view(-1, 1).expand(
            flock_size, self.n_cluster_centers)

        self.create_flocks(params, creator)

    def create_flocks(self, params, creator):
        self.sp_flock = SPFlock(params, creator=creator)
        self.tp_flock = TPFlock(params, creator=creator)

    def run(self, input_data: torch.Tensor, input_context: torch.Tensor = None, input_rewards: torch.Tensor = None):
        forward_mask = self.sp_flock.forward_learn(input_data)
        forward_indices = forward_mask.nonzero()
        self.tp_flock.forward_learn(self.sp_flock.forward_clusters, input_context, input_rewards, forward_mask)
        if self.params.compute_reconstruction:
            self.reconstruct(forward_indices)

        self._assemble_output_context()

    def run_just_sp(self, data: torch.Tensor):
        self.sp_flock.forward_learn(data)

    def run_just_tp(self, data: torch.Tensor, context: torch.Tensor = None, rewards: torch.Tensor = None):
        self.tp_flock.forward_learn(data, context, rewards)

    def reconstruct(self, indices: torch.Tensor = None):
        self.sp_flock.predicted_clusters.copy_(self.tp_flock.action_outputs)
        self.sp_flock.reconstruct(indices)

    def _assemble_output_context(self):
        """Assemble the context for output.

        Context is a 3 dimensional tensor of [flock_size, number_of_context_types, context_size].

        Context is: <SP_output><Pred_clusters for next step>
                    <Rewards><NaNs>
                    <Punishments><NaNs>

        To assemble it, we flatten it down to 2 dims, scatter the outputs, rewards, and punishments then reshape it to
        the desired size.
        """

        # Flatten context so we can scatter over 2 dims
        self.output_context = self.output_context.view(self.params.flock_size, -1)

        # Scatter SP outputs from to the context
        self.output_context.scatter_(dim=0, index=self.index_tensor, src=self.sp_flock.forward_clusters)

        # Scatter the nearest predicted cluster from TP predicted clusters to the context
        start_ind = self.n_cluster_centers * 3
        self.output_context[:, start_ind:].scatter_(dim=0, index=self.index_tensor,
                                                    src=self.tp_flock.passive_predicted_clusters_outputs[:,
                                                        self.seq_lookbehind, :self.n_cluster_centers])

        # Scatter TP action_reward to the context
        start_ind = self.n_cluster_centers
        self.output_context[:, start_ind:].scatter_(dim=0, index=self.index_tensor, src=self.tp_flock.action_rewards)

        # NOTE: Punishments are currently unsupported. The punishment here should always be zero
        start_ind = self.n_cluster_centers * 2
        self.output_context[:, start_ind:].scatter_(dim=0, index=self.index_tensor, src=self.tp_flock.action_punishments)

        # Return the output context to the desired shape of [flock_size, 2, 3, n_cluster_centers]
        self.output_context = self.output_context.view(self.params.flock_size, 2, NUMBER_OF_CONTEXT_TYPES, self.n_cluster_centers)


# TODO: Add ConvTP here
class ConvExpertFlock(ExpertFlock):

    def create_flocks(self, params, creator):
        self.sp_flock = ConvSPFlock(params, creator=creator)
        self.tp_flock = TPFlock(params, creator=creator)
