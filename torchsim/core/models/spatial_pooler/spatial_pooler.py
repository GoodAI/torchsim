import logging
from typing import Union

import torch

from torchsim.core import get_float
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.models.flock import Flock
from torchsim.core.models.spatial_pooler.reconstruction import SPReconstruction
from torchsim.gui.validators import validate_predicate
from .buffer import SPFlockBuffer
from .forward import SPFlockForward, ConvSPStoreToBuffer
from .learning import SPFlockLearning

logger = logging.getLogger(__name__)


class SPFlock(Flock):
    """Defines multiple (i.e. a flock of) spatial poolers.

    Unlike the logical representation of the TA
    (separate experts) each with a SP and TP, the flock is a realisation of multiple
    separate spatial poolers which operate on different data and different clusters, but whose calculations can be
    vectorised easily for speedy calculation.

    The flock handles the forward pass (prediction), and learning phases of the spatial pooler.
    """
    # region Types-Init

    n_cluster_centers: int
    flock_size: int
    input_size: int
    cluster_boost_threshold: int
    buffer_size: int
    batch_size: int
    learning_rate: float
    learning_period: int

    buffer: SPFlockBuffer

    max_boost_threshold: int

    learn_process: SPFlockLearning = None

    def __init__(self, params: ExpertParams, creator: TensorCreator = None):
        """Initialises the flock.

        Args:
            params (ExpertParams): The contain for the parameters which will be used for this flock
            creator (TensorCreator): The creator which will allocate the tensors
        """

        super().__init__(params, creator.device)

        self.n_cluster_centers = params.n_cluster_centers
        self.flock_size = params.flock_size
        self.enable_learning = params.spatial.enable_learning
        float_dtype = get_float(self._device)

        sp_params = params.spatial
        self.input_size = sp_params.input_size
        self.buffer_size = sp_params.buffer_size
        self.batch_size = sp_params.batch_size
        self.learning_period = sp_params.learning_period
        self.max_boost_time = sp_params.max_boost_time
        self.cluster_boost_threshold = sp_params.cluster_boost_threshold
        self.learning_rate = sp_params.learning_rate
        self._boost = sp_params.boost
        self._sampling_method = sp_params.sampling_method

        self.buffer = SPFlockBuffer(creator=creator,
                                    flock_size=self.flock_size,
                                    buffer_size=self.buffer_size,
                                    input_size=self.input_size,
                                    n_cluster_centers=self.n_cluster_centers)

        # The initial clusters are randomised
        self.cluster_centers = creator.zeros((self.flock_size, self.n_cluster_centers, self.input_size),
                                             device=self._device, dtype=float_dtype)
        self.initialize_cluster_centers()

        self.cluster_boosting_durations = creator.full((self.flock_size, self.n_cluster_centers),
                                                       fill_value=self.cluster_boost_threshold,
                                                       device=self._device,
                                                       dtype=creator.int64)

        self.prev_boosted_clusters = creator.zeros((self.flock_size, self.n_cluster_centers), device=self._device,
                                                   dtype=creator.uint8)

        # For holding the targets and deltas of the cluster centers
        self.cluster_center_targets = creator.zeros((self.flock_size, self.n_cluster_centers, self.input_size),
                                                    device=self._device, dtype=float_dtype)
        self.cluster_center_deltas = creator.zeros((self.flock_size, self.n_cluster_centers, self.input_size),
                                                   device=self._device, dtype=float_dtype)
        self.boosting_targets = creator.zeros((self.flock_size, self.n_cluster_centers), device=self._device,
                                              dtype=creator.int64)
        self.tmp_boosting_targets = creator.zeros((self.flock_size, self.n_cluster_centers), device=self._device,
                                                  dtype=creator.int64)

        # Output tensor of cluster center vectors into which to integrate the forward pass stuff
        self.forward_clusters = creator.zeros((self.flock_size, self.n_cluster_centers), device=self._device,
                                              dtype=float_dtype)

        self.predicted_clusters = creator.zeros((self.flock_size, self.n_cluster_centers), device=self._device,
                                                dtype=float_dtype)

        self.current_reconstructed_input = creator.zeros((self.flock_size, self.input_size), device=self._device,
                                                         dtype=float_dtype)
        self.predicted_reconstructed_input = creator.zeros((self.flock_size, self.input_size), device=self._device,
                                                           dtype=float_dtype)

        # How many times did the spatial pooler forward and learning process run
        self.execution_counter_forward = creator.zeros((self.flock_size, 1), device=self._device, dtype=creator.int64)
        self.execution_counter_learning = creator.zeros((self.flock_size, 1), device=self._device, dtype=creator.int64)

    def initialize_cluster_centers(self):
        self.cluster_centers.normal_()

    def _validate_universal_params(self, params: ExpertParams):
        """
        Validate the params which are same in normal and convSP.
        """

        validate_predicate(lambda: params.flock_size > 0)
        validate_predicate(lambda: params.n_cluster_centers > 0)

        spatial = params.spatial
        validate_predicate(lambda: spatial.input_size > 0)
        validate_predicate(lambda: spatial.buffer_size > 0)

        validate_predicate(lambda: spatial.batch_size > 0)
        validate_predicate(lambda: 0 <= spatial.learning_rate <= 1)
        validate_predicate(lambda: spatial.cluster_boost_threshold > 0)
        validate_predicate(lambda: spatial.max_boost_time > 0)
        validate_predicate(lambda: spatial.learning_period > 0)

    def _validate_conv_params(self, params: ExpertParams):
        """
        This needs to be separate because it is overridden by the convSP
        """
        spatial = params.spatial

        validate_predicate(lambda: spatial.buffer_size >= spatial.batch_size)

    def validate_params(self, params: ExpertParams):

        self._validate_universal_params(params)
        self._validate_conv_params(params)

    # endregion

    # region ForwardPass

    @staticmethod
    def _detect_any_difference(data1: torch.Tensor, data2: torch.Tensor):
        # it sometimes crashed on a different sizes of input tensors (not replicable well)
        # if data1.numel() != data2.numel():
        #     return torch.ones(data1.shape)
        sum_of_nans = torch.isnan(data1).type(torch.int32).sum(1)
        sum_of_differences = (data1 != data2).type(torch.int32).sum(1)
        return sum_of_differences != sum_of_nans

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        forward_process, forward_mask = self._determine_forward_process(data)
        self._run_process(forward_process)
        return forward_mask

    def _determine_forward_pass(self, input_data: torch.Tensor, mask: torch.Tensor = None):
        different_mask = self.buffer.inputs.compare_with_last_data(input_data, self._detect_any_difference)
        return different_mask

    # endregion

    # region Learning

    def learn(self, forward_mask: torch.Tensor):
        learn_process = self._determine_learning_process(forward_mask)
        self._run_process(learn_process)

        if learn_process is not None:
            self.learn_process = learn_process

    def _get_learning_buffer(self) -> SPFlockBuffer:
        """Which buffer to use to learn."""
        return self.buffer

    def _determine_learning(self, forward_mask: torch.Tensor):
        learning_buffer = self._get_learning_buffer()
        learning_period_condition = learning_buffer.check_enough_new_data(self.learning_period)
        enough_data_in_buffer_condition = learning_buffer.can_sample_batch(self.batch_size)

        return learning_period_condition * enough_data_in_buffer_condition * forward_mask

    # endregion

    def _determine_forward_process(self, data: torch.Tensor):
        forward_mask = self._determine_forward_pass(data)
        return self._create_forward_process(data, forward_mask.nonzero()), forward_mask

    def _create_forward_process(self, data: torch.Tensor, indices: torch.Tensor):
        # Guard to make sure indices has at least one element for forward pass
        if indices.size(0) == 0:
            # There is no forward pass happening, just return
            return None

        do_subflocking = indices.size(0) != self.flock_size

        forward_process = SPFlockForward(indices,
                                         do_subflocking,
                                         self.buffer,
                                         self.cluster_centers,
                                         self.forward_clusters,
                                         data,
                                         self.n_cluster_centers,
                                         self.input_size,
                                         self.execution_counter_forward,
                                         self._device)

        return forward_process

    def _determine_learning_process(self, forward_mask: torch.Tensor):
        # Start learning
        indices = self._determine_learning(forward_mask).nonzero()
        return self._create_learning_process(indices)

    def _create_learning_process(self, indices: torch.Tensor):
        # Only learn if there is an expert that should learn
        if indices.size(0) == 0:
            # No learning happening.
            return None

        do_subflocking = indices.size(0) != self.flock_size

        learning_process = SPFlockLearning(indices,
                                           do_subflocking,
                                           self.buffer,
                                           self.cluster_centers,
                                           self.cluster_center_targets,
                                           self.cluster_boosting_durations,
                                           self.boosting_targets,
                                           self.cluster_center_deltas,
                                           self.prev_boosted_clusters,
                                           self.n_cluster_centers,
                                           self.input_size,
                                           self.batch_size,
                                           self.cluster_boost_threshold,
                                           self.max_boost_time,
                                           self.learning_rate,
                                           self.learning_period,
                                           self.execution_counter_learning,
                                           self._device,
                                           self._boost,
                                           self._sampling_method)

        return learning_process

    def forward_learn(self, data: torch.Tensor):
        forward_mask = self.forward(data)
        if self.enable_learning:
            self.learn(forward_mask)

        return forward_mask

    def reconstruct(self, indices: Union[torch.Tensor, None]):
        process = self._create_reconstruction_process(indices)
        self._run_process(process)

    def inverse_projection(self, data: torch.Tensor) -> torch.Tensor:
        """Calculates the inverse projection for the given output tensor.

        This is similar to reconstruct(), but operates on the whole flock and the result goes into a newly created
        tensor - it is not stored anywhere on the flock.

        Args:
            data: Tensor matching the shape of forward_clusters, which will be used as the source for the operation.
        """
        if data.shape != self.forward_clusters.shape:
            raise RuntimeError("The provided tensor doesn't match the shape of forward_clusters")

        result = torch.zeros((self.flock_size, self.input_size), dtype=self._float_dtype, device=self._device)

        SPReconstruction.reconstruct(self.flock_size, self.n_cluster_centers, self.cluster_centers, data, result)

        return result

    def _create_reconstruction_process(self, indices: Union[torch.Tensor, None]):

        # If we don't pass any indices in, reconstruct all of them.
        # Otherwise reconstruct only the ones we care about from the forward pass
        if indices is None:
            indices = torch.arange(0, self.flock_size, dtype=torch.int64, device=self._device)
        elif indices.numel() == 0:
            return None

        do_subflocking = indices.size(0) != self.flock_size

        reconstruction_process = SPReconstruction(indices,
                                                  do_subflocking,
                                                  self.cluster_centers,
                                                  self.forward_clusters,
                                                  self.predicted_clusters,
                                                  self.current_reconstructed_input,
                                                  self.predicted_reconstructed_input,
                                                  self.n_cluster_centers,
                                                  self.input_size,
                                                  self._device)

        return reconstruction_process


class ConvSPFlock(SPFlock):
    """Defines a flock of spatial poolers as in the SPFlock, but with differences.

    The SPs share the cluster centers and a big common buffer from which they learn, similarly to a convolutional
    neural network, when a kernel with one matrix of weights is applied on multiple input receptive fields.
    """

    # region Types-Init

    def __init__(self, params: ExpertParams, creator: TensorCreator = torch):
        """Initialises the flock.

        Args:
            params: parameters of the flock, see ExpertParams form default values.
            creator:
        """
        super_params = params.clone()
        super_params.spatial.buffer_size = 1  # these are just internal buffers local to each expert,
        # they do not learn from them.
        super().__init__(super_params, creator)

        super()._validate_universal_params(params)
        self._validate_conv_learning_params(params)

        float_dtype = get_float(self._device)

        # Common buffer where each flock stores data and from which they learn.
        self.common_buffer = SPFlockBuffer(creator=creator,
                                           flock_size=1,
                                           buffer_size=params.spatial.buffer_size,
                                           input_size=self.input_size,
                                           n_cluster_centers=self.n_cluster_centers)

        # The initial clusters are randomised
        self.common_cluster_centers = creator.randn((1, self.n_cluster_centers, self.input_size),
                                                    device=self._device, dtype=float_dtype)

        # Virtual replications of the common cluster centers, mainly for observation purposes.
        self.cluster_centers = self.common_cluster_centers.expand(self.flock_size, self.n_cluster_centers,
                                                                  self.input_size)

        # For keeping track of which clusters are being boosted and for how long
        self.cluster_boosting_durations = creator.full((1, self.n_cluster_centers),
                                                       fill_value=self.cluster_boost_threshold,
                                                       device=self._device,
                                                       dtype=creator.int64)

        self.prev_boosted_clusters = creator.zeros((1, self.n_cluster_centers), device=self._device,
                                                   dtype=creator.uint8)

        # For holding the targets and deltas of the cluster centers
        self.cluster_center_targets = creator.zeros((1, self.n_cluster_centers, self.input_size),
                                                    device=self._device, dtype=float_dtype)
        self.cluster_center_deltas = creator.zeros((1, self.n_cluster_centers, self.input_size),
                                                   device=self._device, dtype=float_dtype)
        self.boosting_targets = creator.zeros((1, self.n_cluster_centers), device=self._device,
                                              dtype=creator.int64)
        self.tmp_boosting_targets = creator.zeros((1, self.n_cluster_centers), device=self._device,
                                                  dtype=creator.int64)

        # There only needs to be one learning counter as only one expert learns in the convolutional case,
        # but we need the expanded version for cluster observer.
        self.common_execution_counter_learning = creator.zeros((1, 1), device=self._device, dtype=creator.int64)
        self.execution_counter_learning = self.common_execution_counter_learning.expand(self.flock_size, 1)

    def _validate_conv_params(self, params: ExpertParams):
        """Validation of the convSP params when seen as individual experts == super_params."""
        spatial = params.spatial
        validate_predicate(lambda: spatial.buffer_size == 1, "ConvSP buffer_size should be 1")

    def _validate_conv_learning_params(self, params: ExpertParams):
        """Validation of the convSP params when seen as one expert == common."""
        spatial = params.spatial

        validate_predicate(lambda: spatial.buffer_size >= params.flock_size,
                           f"In convSP, the common buffer size {{{spatial.buffer_size}}} needs to be at least "
                           f"flock_size {{{params.flock_size}}} for the case that all experts would like to store their "
                           f"data into the buffer at once.")

    def _create_convolutional_storing_process(self, common_buffer: SPFlockBuffer, data: torch.Tensor,
                                              clusters: torch.Tensor, indices: torch.Tensor):
        if indices.size(0) == 0:
            # No learning happening.
            return None
        process = ConvSPStoreToBuffer(indices,
                                      data,
                                      clusters,
                                      common_buffer,
                                      self.n_cluster_centers,
                                      self.input_size,
                                      self._device)

        return process

    def forward_learn(self, data: torch.Tensor):
        forward_mask = self.forward(data)
        common_forward_mask: torch.Tensor = forward_mask.any()  # learn if any expert ran
        if self.enable_learning:
            self.learn(common_forward_mask)

        return common_forward_mask.expand(self.flock_size, 1) # need to recompute reconstruction for all
        # because if at least one learned, the cluster centers might have changed

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        forward_mask = super().forward(data)
        conv_buffer_process = self._create_convolutional_storing_process(self.common_buffer, data,
                                                                         self.forward_clusters, forward_mask.nonzero())
        self._run_process(conv_buffer_process)

        return forward_mask

    def _get_learning_buffer(self) -> SPFlockBuffer:
        return self.common_buffer

    def _create_learning_process(self, indices: torch.Tensor):
        # Only learn if there is an expert that should learn
        if indices.size(0) == 0:
            # No learning happening.
            return None

        learning_process = SPFlockLearning(indices,
                                           False,  # do_subflocking is always false for SP_conv learning
                                           self.common_buffer,
                                           self.common_cluster_centers,
                                           self.cluster_center_targets,
                                           self.cluster_boosting_durations,
                                           self.boosting_targets,
                                           self.cluster_center_deltas,
                                           self.prev_boosted_clusters,
                                           self.n_cluster_centers,
                                           self.input_size,
                                           self.batch_size,
                                           self.cluster_boost_threshold,
                                           self.max_boost_time,
                                           self.learning_rate,
                                           self.learning_period,
                                           self.execution_counter_learning,
                                           self._device,
                                           self._boost,
                                           self._sampling_method)

        return learning_process
