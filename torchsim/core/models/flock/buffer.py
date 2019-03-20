from typing import List, Tuple

from abc import ABC
from contextlib import contextmanager

import torch

from torchsim.core import FLOAT_NAN
from torchsim.core.memory.on_device import OnDevice
from torchsim.core.utils.tensor_utils import change_dim, gather_from_dim
from torchsim.core.models.flock.kernels import buffer_kernels


class Expander(ABC):
    """A base class for tensor expanders.

    An expander handles viewing and broadcasting of values mostly for complex indexing purposes of the buffer.
    """

    def __init__(self, expand_dims):
        """Initializes the expander.

        Args:
            expand_dims: the desired dimensions, usually the same as the dimensions of the tensor to be indexed.
        """

        self._expand_dims = expand_dims
        self._view_dims = self._create_view_dims(expand_dims)

    def _expand(self, tensor: torch.Tensor):
        return tensor.view(self._view_dims).expand(self._expand_dims)

    @staticmethod
    def _create_view_dims(dims):
        # The view dimensions = first dimension + 1 everywhere else (repeated unsqueezing).
        dims = list(dims)
        return tuple(dims[:1] + ([1] * (len(dims) - 1)))


class StaticItemCountExpander(Expander):
    """And expander which has static item count instead of buffer_size."""

    def __init__(self, dims: Tuple[int, ...], item_count: int = None):
        if item_count is not None:
            dims = change_dim(dims, index=1, value=item_count)
        super().__init__(dims)

    def expand(self, tensor: torch.Tensor):
        return self._expand(tensor)


class VariableItemCountExpander(Expander):
    """And expander which has static item count instead of buffer_size."""

    def expand(self, tensor: torch.Tensor, item_count: int):
        self._view_dims = change_dim(self._view_dims, index=1, value=item_count)
        self._expand_dims = change_dim(self._expand_dims, index=1, value=item_count)
        return self._expand(tensor)


class CurrentValueNotStoredException(Exception):
    """Raised when not all storages were stored into during buffer.next_step()."""
    pass


class BufferStorage(OnDevice):
    """A storage stores items of the same size in the buffer.

    The storage provides methods for storing a single item for all experts, sampling of items for e.g. learning
    purposes and detection whether the provided item is different from the item stored at the buffer's current pointer.
    """
    _buffer: 'Buffer'

    stored_data: torch.Tensor
    _steps_to_write: int

    def __init__(self, creator, name, buffer: 'Buffer', dims: Tuple[int, ...], dtype, force_cpu: bool = False):
        """Initializes the storage.

        Args:
            name (str): name of the storage for debugging/error handling purposes
            buffer (Buffer): a reference to the owning buffer instance
            dims (object): (flock_size, buffer_size, data_size)
            dtype (str): type of the values in the tensor
        """
        device = 'cpu' if force_cpu else creator.device
        super().__init__(device)

        self._force_cpu = force_cpu
        self.dims = dims
        self.name = name
        dtype = dtype or self._float_dtype
        if force_cpu and dtype == torch.float16:
            raise ValueError(f"Buffer: {name} is on CPU, but has erroneously been defined to use 16-bit values.")

        self._buffer = buffer
        self.stored_data = creator.full(dims, fill_value=FLOAT_NAN, dtype=dtype, device=self._device)

        self._all_items_expander = StaticItemCountExpander(self.dims)
        self._single_item_expander = StaticItemCountExpander(self.dims, item_count=1)
        self._batch_expander = VariableItemCountExpander(self.dims)
        self._steps_to_write = 0

        if self._force_cpu:
            self.stored_data = self.stored_data.pin_memory()

    @property
    def current_ptr(self):
        return self._buffer.current_ptr

    def store(self, data: torch.Tensor):
        """Stores 'data' into the current position in the buffer.

        Args:
            data (torch.Tensor): The data to be stored [flock_size, data_size]
        """
        if self._buffer.flock_indices is None:
            data = torch.unsqueeze(data, dim=1)
            data_pointers = self._single_item_expander.expand(self._buffer.current_ptr.to(self._device))
            self.stored_data.scatter_(1, data_pointers, data.to(self._device).type(self._float_dtype))
        else:
            if self._force_cpu:
                flock_indices = self._buffer.flock_indices
                row_indices = self._buffer.current_ptr[flock_indices].to(self._device)
                self.stored_data[flock_indices.to(self._device), row_indices] = data.type(self._float_dtype).to(self._device)

            else:
                flock_indices = self._buffer.flock_indices
                # TODO: this could be also part of the kernel below
                row_indices = gather_from_dim(self._buffer.current_ptr, flock_indices, dim=0)

                flock_size = flock_indices.size(0)

                buffer_kernels.buffer_store(self.stored_data.view(self.dims[:2] + (-1,)),
                                            flock_indices,
                                            row_indices,
                                            data.view(flock_size, -1),
                                            data.numel()//flock_size,
                                            flock_size)

        self._steps_to_write -= 1

    def get_stored_data(self):
        if self._buffer.flock_indices is None:
            return self.stored_data.type(self._buffer._float_dtype).to(self._buffer._device)
        else:
            return gather_from_dim(self.stored_data, self._buffer.flock_indices.to(self._device), 0).type(self._buffer._float_dtype).to(self._buffer._device)

    def set_stored_data(self, data):
        if self._buffer.flock_indices is None:
            self.stored_data.copy_(data.to(self._device))
        else:
            self.stored_data[self._buffer.flock_indices.to(self._device)] = data.to(self._device)

    def store_batch(self, data: torch.Tensor):
        """Stores a whole batch of data.

        It expects on input a tensor with shape [flock_size == dims[0], batch_size, dims[1:]].

        This batch_size has nothing to do with the batch_size used for sampling from the buffer!
        It cannot be larger than the buffer_size.
        """

        if self._buffer.flock_indices is None:
            batch_size = data.size(1)
            assert batch_size <= self._buffer.buffer_size, "Stored batch must be at most as large as the buffer."

            start = self._buffer.current_ptr - batch_size + 1
            selector = start.view(-1, 1).expand(self._buffer.flock_size, batch_size)
            selector = (selector + self._buffer.batching_tensor[0:batch_size]) % self._buffer.buffer_size
            selector = self._batch_expander.expand(selector, batch_size)

            self.stored_data.scatter_(dim=1, index=selector.to(self._device), src=data.to(self._device))
        else:
            raise NotImplementedError

        self._steps_to_write -= batch_size

    def invalidate_current(self, steps: int):
        """Mark the storage as not written into (expecting `steps` amount of entries)."""
        self._steps_to_write = steps

    def are_all_current_values_stored(self):
        """Check that the correct amount of data was written into the storage."""
        return self._steps_to_write == 0

    def sample_forward_batch(self, batch_size: int, out: torch.Tensor):
        self._sample_batch(batch_size, out)

    def sample_contiguous_batch(self, batch_size: int, out: torch.Tensor):
        self._sample_batch(batch_size, out)

        self._buffer.reset_data_since_last_sample()

    def _sample_batch(self, batch_size: int, out: torch.Tensor):
        """Samples a batch from the buffer.

        The buffer is read backward so that the first inputs are the chronologically oldest.

        Args:
            out (torch.Tensor): Tensor of shape (flock_size, batch_size, ...other dimensions) for the result batch
            batch_size (int): The size of the sample we wish to draw
        """
        out_device = out.device
        out_dtype = out.dtype
        assert batch_size <= self._buffer.buffer_size, \
            f"Batch size {batch_size} cannot be greater than buffer size {self._buffer.buffer_size}."

        start = self._buffer.current_ptr - batch_size + 1
        selector = start.view(-1, 1).expand(self._buffer.flock_size, batch_size)
        selector = (selector + self._buffer.batching_tensor[0:batch_size]) % self._buffer.buffer_size
        selector = self._batch_expander.expand(selector, batch_size).to(self._device)

        if self._buffer.flock_indices is None:
           vals = torch.gather(self.stored_data, dim=1, index=selector)

        else:
            # TODO: this is a good place for a custom kernel - 3 gathers
            selector = gather_from_dim(selector, self._buffer.flock_indices.to(self._device), dim=0)
            vals = torch.gather(gather_from_dim(self.stored_data, self._buffer.flock_indices.to(self._device), dim=0),
                                dim=1, index=selector)

        out.copy_(vals.type(out_dtype).to(out_device))

    def compare_with_last_data(self, data: torch.Tensor, comparator):
        """Compares the passed in data to that stored in the locations of the buffer pointed to by the data pointers.

        Args:
            data (torch.Tensor): The new inputs to experts [0, 1, ..., n]

        Returns:
            A binary tensor indicating inputs are different from the current position of the buffer.
        """
        ret_device = data.device
        ret_dtype = data.dtype
        # The current pointers look at the last thing placed into the buffer
        pointers = self._single_item_expander.expand(self._buffer.current_ptr).to(self._device)
        data_at_pointers = torch.gather(self.stored_data, 1, pointers).squeeze(dim=1)

        # TODO: (Future) Valid only under pytorch1.0
        # return torch.eq(data, data_at_pointers).all(dim=1)

        return comparator(data, data_at_pointers.type(ret_dtype).to(ret_device))

    def reorder(self, indices: torch.Tensor):
        """Reorders the data along the dimension 2.

         If we want more dimensions, the easiset way would probably be to reshape everything to
         flock_size, buffer_size, -1 and then back.
        """
        max = self.dims[2]
        # identify the columns which are not present - they will be replaced by nans
        newly_frequent_sequences = indices >= max

        # point to the nan column
        indices_with_nans = indices.clone().to(self._device)
        indices_with_nans[newly_frequent_sequences] = max

        if self._buffer.flock_indices is None:

            nan_column = torch.full((self._buffer.flock_size, self._buffer.buffer_size, 1), fill_value=FLOAT_NAN,
                                    dtype=self._float_dtype,
                                    device=self._device)

            original_data_with_nan_column = torch.cat([self.stored_data, nan_column], dim=2)

            torch.gather(input=original_data_with_nan_column, dim=2,
                         index=indices_with_nans.unsqueeze(dim=1).expand(self.dims), out=self.stored_data)
        else:
            flock_indices = self._buffer.flock_indices.to(self._device)
            new_flock_size = flock_indices.numel()
            new_dims = change_dim(self.dims, index=0, value=new_flock_size)
            new_indices_with_nans = indices_with_nans.unsqueeze(dim=1).expand(new_dims)

            nan_column = torch.full((new_flock_size, self._buffer.buffer_size, 1),
                                    fill_value=FLOAT_NAN, dtype=self._float_dtype, device=self._device)

            original_data_with_nan_column = torch.cat([self.stored_data[flock_indices], nan_column], dim=2)

            self.stored_data[flock_indices] = torch.gather(original_data_with_nan_column, dim=2,
                                                           index=new_indices_with_nans)


class Buffer(OnDevice, ABC):
    """A base class for buffers.

    Handles subflocking using the provided indices. If no subflocking should be done, updates the whole tensors.
    When a new item should be written in the buffer, you should call next_step() and then store an item in all
    of the storages of the buffer.
    """
    _storages: List[BufferStorage]
    buffer_size: int
    flock_size: int
    current_ptr: torch.Tensor
    data_since_last_sample: torch.Tensor
    batching_tensor: torch.Tensor
    flock_indices: torch.Tensor

    def __init__(self, creator, flock_size: int, buffer_size: int):
        """Initializes the buffer."""
        super().__init__(creator.device)

        self._creator = creator
        self.flock_size = flock_size
        self.buffer_size = buffer_size

        self._storages = []

        self.current_ptr = creator.full((flock_size,), fill_value=self.buffer_size - 1, device=self._device,
                                        dtype=creator.int64)

        # How many new data entries have been written since the last sampling
        self.data_since_last_sample = creator.zeros((flock_size,), device=self._device, dtype=creator.int64)

        # How many entries since the creation of the buffer
        self.total_data_written = creator.zeros((flock_size,), device=self._device, dtype=creator.int64)

        # Preallocate batching tensor for sampling (maximum batch_size will be buffer_size) - it is ok to preallocate
        # for the maximum size because the size is constant and negligible. It is better to have everything preallocated
        # to be able to asses the buffer size and thus how large flock will fit the memory.
        self.batching_tensor = creator.arange(0, buffer_size, device=self._device, dtype=creator.int64)

        # Set by the next_step context manager if there are only specific indices that should be written to
        self.flock_indices = None

    def reset_data_since_last_sample(self):
        if self.flock_indices is None:
            self.data_since_last_sample.fill_(0)
        else:
            self.data_since_last_sample.index_fill_(dim=0, index=self.flock_indices, value=0)
            # optimized code for:
            # self.data_since_last_sample[self.flock_indices] = 0

    def _create_storage(self, name, dims: Tuple[int, ...], dtype=None, force_cpu: bool = False) -> BufferStorage:
        """Creates a storage with the given name, dimensions and dtype.

        Note that dims must be of the shape (flock_size, buffer_size, data_size). Individual storages' dims only differ
        in the data_size bit.
        """
        buffer_storage = BufferStorage(self._creator, name, self, dims, dtype, force_cpu)
        self._storages.append(buffer_storage)
        return buffer_storage

    def _increase_pointers(self, steps=1):
        """A new item will be written, move the pointers forward."""
        if self.flock_indices is None:
            self.current_ptr = (self.current_ptr + steps) % self.buffer_size
            self.total_data_written += steps
            self.data_since_last_sample += steps
        else:
            steps_tensor = torch.full(self.flock_indices.size(), fill_value=steps, device=self._device,
                                      dtype=torch.int64)
            # self.current_ptr[self.flock_indices] += steps
            self.current_ptr.scatter_add_(dim=0, index=self.flock_indices, src=steps_tensor)
            self.current_ptr = self.current_ptr % self.buffer_size
            # self.total_data_written[self.flock_indices] += steps
            self.total_data_written.scatter_add_(dim=0, index=self.flock_indices, src=steps_tensor)
            # self.data_since_last_sample[self.flock_indices] += steps
            self.data_since_last_sample.scatter_add_(dim=0, index=self.flock_indices, src=steps_tensor)

    def can_sample_batch(self, batch_size: int) -> torch.Tensor:
        """Checks which expert buffer contains enough data to sample `batch_size` lines."""
        return self.total_data_written >= batch_size

    def check_enough_new_data(self, period) -> torch.Tensor:
        """Checks that enough data has been put into the buffer since the last sample."""
        return self.data_since_last_sample >= period

    @contextmanager
    def next_step(self):
        """A context manager for writing new items into the buffer.

        This moves the pointers, invalidates the current items, and upon exiting it checks whether all storages
        were written into.
        """
        self._increase_pointers()
        for storage in self._storages:
            storage.invalidate_current(1)
        yield
        for storage in self._storages:
            if not storage.are_all_current_values_stored():
                raise CurrentValueNotStoredException(f"Did not store into {storage.name} or stored twice.")

    @contextmanager
    def next_n_steps(self, steps):
        """A context manager for writing new items into the buffer.

        This moves the pointers, invalidates the current items, and upon exiting it checks whether all storages
        were written into.
        """
        self._increase_pointers(steps)
        for storage in self._storages:
            storage.invalidate_current(steps)
        yield
        for storage in self._storages:
            if not storage.are_all_current_values_stored():
                raise CurrentValueNotStoredException(f"Did not store into {storage.name} or stored incorrect amount.")

    def set_flock_indices(self, indices: torch.Tensor):
        self.flock_indices = indices.view(-1).to(self._device)

    def unset_flock_indices(self):
        self.flock_indices = None
