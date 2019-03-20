from functools import reduce
from typing import List, Optional, Any

import torch
from abc import ABC, abstractmethod


def get_dims(items: List):
    dims = []
    while type(items) == list:
        dims.append(len(items))
        items = items[0]

    return dims


class TensorCreator(ABC):
    float32 = float = torch.float32
    float64 = double = torch.float64
    float16 = half = torch.float16
    uint8 = torch.uint8
    int8 = torch.int8
    int16 = short = torch.int16
    int32 = int = torch.int32
    int64 = long = torch.int64
    device: str

    def __init__(self, device):
        self.device = device

    @staticmethod
    @abstractmethod
    def tensor(data, dtype=None, device=None, requires_grad=False) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def zeros(*dims, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None,
             requires_grad=False) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def ones(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def empty(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def cat(seq, dim=0, out=None) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def arange(start, end, step=1, out=None, dtype=None, layout=torch.strided, device=None,
               requires_grad=False) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def rand(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def randn(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def randint(low, high, size, out=None, dtype=None, layout=torch.strided, device=None,
                requires_grad=False) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> torch.Tensor:
        pass


class TensorSurrogate:
    """A tensor surrogate is used in update_memory_blocks (the iterative tensor size determination algorithm).

    Just the shape is important.

    Note that we probably don't need to do any checks here, because they will be done as part of normal torch
    initialization.
    """
    device: str
    dtype: Any  # TODO specify type
    shape: torch.Size

    def __init__(self, dims, dtype=None, device: Optional[str] = None):
        self.shape = torch.Size(dims)
        self.dtype = dtype
        self.device = device

    @staticmethod
    def _infer_dimension(shape, numel):
        result_shape = list(shape)
        if -1 not in shape:
            return result_shape

        inferred_idx = None
        numel_rest = 1
        for i, dim in enumerate(shape):
            if dim == -1:
                # No check for multiple '-1's.
                inferred_idx = i
            else:
                numel_rest *= dim

        # No checks - assume it's divisible.
        inferred_dim = numel // numel_rest
        result_shape[inferred_idx] = inferred_dim

        return torch.Size(result_shape)

    @staticmethod
    def _process_dims_args(dims):
        if len(dims) == 1:
            first = dims[0]
            t = type(first)
            if t in (tuple, list, torch.Size):
                dims = first

        return dims

    def view(self, *target_shape):
        target_shape = self._process_dims_args(target_shape)

        return TensorSurrogate(self._infer_dimension(target_shape, self.numel()), self.dtype, self.device)

    def expand(self, *target_shape):
        target_shape = list(self._process_dims_args(target_shape))

        # No checks done here.
        for i in range(len(target_shape)):
            if target_shape[i] == -1:
                target_shape[i] = self.shape[i]

        return TensorSurrogate(target_shape, self.dtype, self.device)

    def numel(self) -> int:
        """Number of elements of this tensor."""
        return reduce(lambda a, dim: a * dim, self.shape)

    def size(self):
        return self.shape

    def __setitem__(self, key, value):
        """Setting particular values does nothing."""
        pass

    def contiguous(self):
        return self

    def normal_(self):
        pass

    def dim(self):
        return len(self.shape)

    def reshape(self, *target_shape):
        return self.view(*target_shape)

    def gather(self, dim: int, index: 'TensorSurrogate'):
        if dim > self.dim() - 1:
            raise RuntimeError
        # Check that all dims except for 'dim' have the same size
        for i in range(self.dim()):
            if i != dim and self.shape[i] != index.shape[i]:
                raise RuntimeError
        return index

    def long(self):
        return self

    def copy_(self, src):
        self.shape = src.shape
        self.device = src.device
        self.dtype = src.dtype

    def unsqueeze(self, dim):
        old_shape = list(self.shape)

        if dim < 0:
            dim = len(old_shape) + (dim + 1)

        new_shape = old_shape[:dim] + [1] + old_shape[dim:]

        return TensorSurrogate(new_shape, self.dtype, self.device)

    def fill_(self, value):
        return self

    # We might add other stuff here later, like indexing.


# noinspection PyTypeChecker
class MeasuringCreator(TensorCreator):
    """A TensorCreator which creates instances of TensorSurrogate instead of torch.Tensor.

    The surrogates only require dimensions.
    """

    def __init__(self):
        super().__init__("cpu")

    @staticmethod
    def _process_star_dims(dims):
        if type(dims[0]) in (tuple, list, torch.Size):
            return dims[0]

        return dims

    @staticmethod
    def tensor(data, dtype=None, device=None, requires_grad=False) -> torch.Tensor:
        return TensorSurrogate(get_dims(data), dtype, device)

    @staticmethod
    def zeros(*dims, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> torch.Tensor:
        dims = MeasuringCreator._process_star_dims(dims)
        return TensorSurrogate(dims, dtype, device)

    @staticmethod
    def full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None,
             requires_grad=False) -> torch.Tensor:
        return TensorSurrogate(size, dtype, device)

    @staticmethod
    def ones(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> torch.Tensor:
        sizes = MeasuringCreator._process_star_dims(sizes)
        return TensorSurrogate(sizes, dtype, device)

    @staticmethod
    def empty(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> torch.Tensor:
        sizes = MeasuringCreator._process_star_dims(sizes)
        return TensorSurrogate(sizes, dtype, device)

    @staticmethod
    def cat(seq, dim=0, out=None) -> torch.Tensor:
        if out is not None:
            # The result must already have the same dimensions
            return

        dims = list(seq[0].shape)
        dims[dim] = sum(map(lambda tensor: tensor.shape[dim], seq))
        return TensorSurrogate(dims, seq[0].dtype, seq[0].device)

    @staticmethod
    def eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
        m = m or n
        return TensorSurrogate((n, m), dtype, device)

    @staticmethod
    def arange(start, end, step=1, out=None, dtype=None, layout=torch.strided, device=None,
               requires_grad=False) -> torch.Tensor:
        if out is not None:
            # The result must already have the same dimensions
            return

        dim_size = (end + 1 - start) // step
        return TensorSurrogate((dim_size,), dtype, device)

    @staticmethod
    def rand(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> torch.Tensor:
        sizes = MeasuringCreator._process_star_dims(sizes)
        return TensorSurrogate(sizes, dtype, device)

    @staticmethod
    def randn(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> torch.Tensor:
        sizes = MeasuringCreator._process_star_dims(sizes)
        return TensorSurrogate(sizes, dtype, device)

    @staticmethod
    def randint(low, high, size, out=None, dtype=None, layout=torch.strided, device=None,
                requires_grad=False) -> torch.Tensor:
        return TensorSurrogate(size, dtype, device)

    @staticmethod
    def zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> torch.Tensor:
        return TensorSurrogate(input.size(), dtype, device)


def generate_torch_delegators(cls):
    class Wrapped(cls):
        pass

    # Go through all abstract methods and delegate their calls to torch.
    for name in cls.__abstractmethods__:
        setattr(Wrapped, name, staticmethod(getattr(torch, name)))

    Wrapped.__abstractmethods__ = None

    return Wrapped


# noinspection PyAbstractClass
@generate_torch_delegators
class AllocatingCreator(TensorCreator):
    """A creator which passes all TensorCreator API calls on to the torch module.

    The abstract methods from TensorCreator are automatically generated by the decorator, except for methods explicitly
    implemented here in this class.
    """
