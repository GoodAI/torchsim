from typing import Tuple

import torch

from torchsim.core.models.flock.buffer import BufferStorage, Buffer


class NetworkFlockBuffer(Buffer):
    """Defines a circular buffer for a flock of neural networks.

    It has two storages: for inputs (flock_size, buffer_size, input_size) and for outputs.

    Then there are learning_coefficients (flock_size, 1) determining how much each network should learn the particular IO.
    """

    inputs: BufferStorage
    targets: BufferStorage
    learning_coefficients: BufferStorage

    _delay_used: bool

    def __init__(self,
                 creator,
                 flock_size: int,
                 buffer_size: int,
                 input_shape: Tuple,
                 target_shape: Tuple,
                 delay_used: bool):
        """Initialize the buffer

        Args:
            flock_size (int): Number of networks in the flock
            buffer_size (int): Number of elements that can be stored in the buffer before rewriting occurs
            input_shape (Tuple): The shape of the inputs
            target_shape (Tuple): The shape of the target
            delay_used (bool): whether any of the fields is delayed (then the can_sample_batch waits one more step)
        """
        super().__init__(creator, flock_size, buffer_size)

        self._delay_used = delay_used
        self.inputs = self._create_storage("inputs", (flock_size, buffer_size, *input_shape), force_cpu=False)
        self.targets = self._create_storage("targets", (flock_size, buffer_size, *target_shape))
        self.learning_coefficients = self._create_storage("coefficients", (flock_size, buffer_size, 1))

    def store(self, inputs: torch.Tensor, targets: torch.Tensor, coefficients: torch.Tensor):
        """
        Stores 'inputs' and 'targets' into the current position in the buffer, one for each network in the flock.
        For each network in the flock there is one coefficient with which this sample should be learned.
        """

        self.flock_indices = None  # or a 1D tensor with indices where to write

        with self.next_step():
            self.inputs.store(inputs)
            self.targets.store(targets)
            self.learning_coefficients.store(coefficients)

    def can_sample_batch(self, batch_size: int) -> bool:
        """Return true if all the networks in the flock have enough data in the buffer.

        In this case, either none of the networks has enough data or all of them.
        Note that the data in one of the BufferStorages is delayed, we should wait one more step for it
        (because it contains invalid values in the first step).
        """

        if not self._delay_used:
            min_can_sample: torch.Tensor = super().can_sample_batch(batch_size).min()
        else:
            min_can_sample: torch.Tensor = (self.total_data_written - 1 >= batch_size).min()

        return min_can_sample.item() > 0

    def sample_learning_batch(self, batch_size: int,
                              inputs: torch.tensor,
                              targets: torch.Tensor,
                              coefficients: torch.Tensor):
        """Samples a batch from the buffer using the LAST_N sampling method

        The buffer is read backward so that the first inputs are the chronologically oldest.

        Args:
            coefficients: Tensor of shape (flock_size, batch_size, 1) for the coefficients
            targets: Tensor for targets, shape (flock_size, batch_size, ...other dimensions)
            inputs: similar to the targets
            batch_size (int): The size of the sample we wish to draw
        """

        self.flock_indices = None
        self.inputs.sample_contiguous_batch(batch_size, inputs)
        self.targets.sample_contiguous_batch(batch_size, targets)
        self.learning_coefficients.sample_contiguous_batch(batch_size, coefficients)

