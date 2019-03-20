import gc
import sys
from typing import Union, Tuple, Any, Optional, Sequence
import numpy as np

import torch
from contextlib import ContextDecorator, contextmanager
import logging
from functools import wraps
import warnings

from prettytable import PrettyTable

from torchsim.core import get_float, SMALL_CONSTANT


@contextmanager
def tensor_allocation_monitor(name):
    """Decorator for analyzing which tensors are created and which are put out of scope.

    Use only for debugging.

    Args:
        name: Name of the function which it is analyzing. Will print the name on the report.
    """
    before = set([(id(obj), obj.dtype, obj.size(), len(gc.get_referrers(obj)) - 1) for obj in gc.get_objects() if torch.is_tensor(obj) and obj._base is None])
    yield
    gc.collect()
    after = set([(id(obj), obj.dtype, obj.size(), len(gc.get_referrers(obj)) - 1) for obj in gc.get_objects() if torch.is_tensor(obj) and obj._base is None])

    set_minus_alloc = after - before
    set_minus_dealloc = before - after

    if len(set_minus_alloc) > 0:
        print(f"\tFunction: {name} allocated tensors")
        for t in set_minus_alloc:
            print("\t\t", t)

    if len(set_minus_dealloc) > 0:
        print(f"\t****Function: {name} has tensors which go out of scope.****")
        for t in set_minus_dealloc:
            print("\t\t", t)

    if len(set_minus_dealloc) > 0 or len(set_minus_alloc) > 0:
        print("\n")


class log_torch_memory(ContextDecorator):
    """Logs PyTorch-allocated and used memory on the GPU."""

    def __enter__(self):
        self.logger.debug(f"[Enter] *** cuda heap: {torch.cuda.max_memory_allocated()/(1024 ** 2)}MB *** "
                          f"cuda heap used {torch.cuda.memory_allocated()/(1024 ** 2)}MB.")
        return self

    def __call__(self, f):
        self.logger = logging.getLogger(f.__repr__().split(" ")[1])

        @wraps(f)
        def wrapper(*args, **kw):
            with self:
                return f(*args, **kw)

        return wrapper

    def __exit__(self, *exc):
        self.logger.debug(f"[Exit] *** cuda heap: {torch.cuda.max_memory_allocated()/(1024 ** 2)}MB *** "
                          f"cuda heap used {torch.cuda.memory_allocated()/(1024 ** 2)}MB.")
        return False


def change_dim(dims, index, value):
    """Changes the dimension at index to value."""
    dims = list(dims)
    dims[index] = value
    return tuple(dims)


def multi_unsqueeze(tensor: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
    """Unsqueezes the tensor to match number of dimensions in dims."""
    for k in dims:
        tensor = tensor.unsqueeze(k)
    return tensor


def normalize_probs(tensor: torch.Tensor, dim: int, add_constant=False):
    """Normalizes the given probabilities.

    If add_constant is True, this will add a small constant to the whole tensor. This is used when we don't
    want any probability to be 0.

    The vector is normalized so that it sums up to 1.
    """
    if add_constant:
        tensor = add_small_constant(tensor, SMALL_CONSTANT)

    summed = torch.sum(tensor, dim=dim, keepdim=True)

    return tensor / summed


def normalize_probs_(tensor: torch.Tensor, dim: int, add_constant=False):
    """Normalizes probabilities in-place."""
    if add_constant:
        tensor += SMALL_CONSTANT

    summed = torch.sum(tensor, dim=dim, keepdim=True)
    tensor /= summed


def move_probs_towards_50(tensor: torch.Tensor) -> torch.Tensor:
    """Moves each value towards 0.5.

    Moves less the closer it already is to 0.5.
    Expects all values of `tensor` to lie in the interval [0, 1]
    """
    diff = tensor - 0.5
    return tensor - diff * SMALL_CONSTANT * 2.0


def move_probs_towards_50_(tensor: torch.Tensor):
    """In-place version of move_probs_towards_50."""
    torch.add(tensor, alpha=-SMALL_CONSTANT * 2.0, other=(tensor - 0.5), out=tensor)


def add_small_constant_(tensor: torch.Tensor, constant: float):
    tensor += constant


def add_small_constant(tensor: torch.Tensor, constant: float) -> torch.Tensor:
    return tensor + constant


def negate(tensor: torch.Tensor) -> torch.Tensor:
    """Computes logical negation of a tensor.

    Args:
        tensor: Tensor representing logical values: 0 means False, anything else means True.

    Returns:
        Negation of the `tensor`, i.e., 1 on all places where the `tensor` had non-zero values and vice versa.

    """
    return tensor == 0


def same(tensor_1: torch.Tensor, tensor_2: torch.Tensor, eps=None):
    """Returns True if the tensors have identical values with eps tolerance.

    If eps is None, the values have to match exactly.
    """
    # Check sizes.
    if tensor_1.shape != tensor_2.shape:
        return False

    not_nans1 = negate(torch.isnan(tensor_1))
    not_nans2 = negate(torch.isnan(tensor_2))

    # Check nans.
    if (not_nans1 != not_nans2).any():
        return False

    if eps is None:
        return (tensor_1[not_nans1] == tensor_2[not_nans2]).all()
    else:
        if eps > 1:
            warnings.warn("eps is intended to be a small number in the form '1e-n'.", stacklevel=2)
        return ((tensor_1[not_nans1] - tensor_2[not_nans2]).abs() < eps).all()


def weighted_sum_(tensor_a: torch.Tensor,
                  weight_a: Union[torch.Tensor, float],
                  tensor_b: torch.Tensor,
                  weight_b: Union[torch.Tensor, float],
                  output: torch.Tensor):
    output.copy_(weighted_sum(tensor_a=tensor_a,
                              weight_a=weight_a,
                              tensor_b=tensor_b,
                              weight_b=weight_b))


def weighted_sum(tensor_a: torch.Tensor,
                 weight_a: Union[torch.Tensor, float],
                 tensor_b: torch.Tensor,
                 weight_b: Union[torch.Tensor, float]) -> torch.Tensor:

    return weight_a * tensor_a + weight_b * tensor_b


def kl_divergence(input_p: torch.Tensor,
                  input_q: torch.Tensor,
                  output: torch.Tensor,
                  dim):
    """Computes KL-divergence between two discrete distributions P and Q along the `dim` dimension.

    D_KL(P || Q) = sum_i( P(i) log (P(i) / Q(i)))
    Expects P and Q to be normalized along the `dim` dimension, i.e., to have sum 1 and have all values > 0
    """

    # Compute KL-divergence between the distribution based on clusters and priors and
    # distribution based on clusters, prior + each context separately
    # TODO (?): write using pytorch KL divergence? Is it faster? Does it compute exactly what we want (nans, etc.)?
    # Note, torch kl_div(p, q) = sum_i( q_i * ( log(q_i) - p_i )). i.e. p and q are swapped and log(p_i) is not computed
    # context_informativeness.copy_(
    #     torch.nn.functional.kl_div(normalized_baseline,
    #                                normalized_seq_likelihoods_for_each_context,
    #                                reduction='none').sum(dim=1))

    divergence = input_p * torch.log(input_p / input_q)
    divergence.masked_fill_(torch.isnan(divergence), 0)

    output.copy_(divergence.sum(dim=dim))


def detect_qualitative_difference(data1: torch.Tensor, data2: torch.Tensor):
    return data1.argmax(dim=1) != data2.argmax(dim=1)


class WrongTensorShapeException(Exception):
    pass


def check_shape(first, second):
    if first != second:
        raise WrongTensorShapeException(f"The input shape is different than expected ({first} vs {second}")


def view_dim_as_dims(tensor: torch.Tensor, shape: Tuple[int, ...], dim: int = -1):
    """Replaces tensor with a view, deriving the shape from the tensor by replacing the `dim` by `shape`.

    Args
        tensor - Tensor to be reshaped
        shape - Dimensions that will replace selected dimension
        dim - Dimension to be replaced. Negative values are supported (to index from the end)

    Examples
        tensor.shape=(1, 4, 5), shape=(2, 2), dim=-2, result -> (1, 2, 2, 5)
    """
    original_dims = tensor.size()
    view_dims = original_dims[:dim] + shape
    # add the tail just if there is one (this negative indexing needs to be handled separately
    if dim != -1:
        view_dims += original_dims[(dim + 1):]

    return tensor.view(view_dims)


def gather_from_dim(source: torch.Tensor, indices: torch.Tensor, dim: int = 0):
    """Gathers values from `source` along the dimension `dim`.

    The values to gather are specified in the vector `indices`, which should be one-dimesional
    """
    # TODO: This check is too costly, enable it only in case we have asserts called during debugging only
    # if indices.max() > source.size(dim):
    #     raise ValueError(f"Max index ({indices.max()} cannot be larger than the number of elements {source.size(dim)}"
    #                      f" across the chosen dim {dim}")

    if indices.dim() > 1:
        raise ValueError(f"Expected a vector of indices, but got something with {indices.dim()}"
                         f" dims.")

    # index_select
    return source.index_select(dim, indices)

    # TODO: Here are keep some alternative version for which were empirically slower. How fast would be a custom kernel?
    # # V2: original gather_from_dims
    # n_indices = indices.numel()
    #
    # view_dims = [1] * source.dim()
    # view_dims[dim] = n_indices
    #
    # expand_dims = list(source.size())
    # expand_dims[dim] = n_indices
    #
    # indices = indices.view(view_dims).expand(expand_dims)
    #
    # return torch.gather(source, index=indices, dim=dim)

    # # V3: advanced indexing
    # slices = [slice(None)] * dim
    # slices += [indices]
    # return source[slices]

    # # V4: permute dimensions (so that the dim is at zeroth position) -> advanced indexing -> unpermute dimensions
    # # permute then smart sampling over first dim
    # perm_ind = list(range(source.dim()))
    # del perm_ind[dim]
    # perm_ind = [dim] + perm_ind
    #
    # tmp = source.permute(*perm_ind)
    # filtered = tmp[indices]
    # return filtered.permute(*perm_ind)


def scatter_(source: torch.Tensor, destination: torch.Tensor, indices: torch.Tensor, dim: int = 0):
    # TODO: why is it here, indices should be always in one format (preferably 1D vector)
    indices = indices.view(-1)

    destination.index_copy_(dim, indices, source)

    # # Original version
    # view_dims = indices.size() + (1,) * (source.dim() - indices.dim())
    # expand_dims = source.size()
    #
    # indices = indices.view(view_dims).expand(expand_dims)
    #
    # destination.scatter_(dim=dim, index=indices, src=source)


def average_abs_values_to_float(input_tensor: torch.Tensor) -> float:
    """Compute average of the tensor.

    Args:
        input_tensor: tensor on 'cpu' or 'cuda'

    Returns:
        Sum of absolute values divided by number of elements.
    """
    total_delta = input_tensor.abs_().sum().to('cpu').item()
    return total_delta / input_tensor.numel()

def id_to_one_hot(data: torch.Tensor, vector_len: int, dtype: Optional[torch.dtype] = None):
    """Converts ID to one-hot representation.

    Each element in `data` is converted into a one-hot-representation - a vector of
    length vector_len having all zeros and one 1 on the position of value of the element.

    Args:
        data: ID of a class, it must hold for each ID that 0 <= ID < vector_len
        vector_len: length of the output tensor for each one-hot-representation
        dtype: data type of the output tensor

    Returns:
        Tensor of size [data.shape[0], vector_len] with one-hot encoding.
        For example, it converts the integer cluster indices of size [flock_size, batch_size] into
        one hot representation [flock_size, batch_size, n_cluster_centers].
    """
    device = data.device
    dtype = dtype or get_float(device)

    data_a = data.view(-1, 1)
    n_samples = data_a.shape[0]
    output = torch.zeros(n_samples, vector_len, device=device, dtype=dtype)
    output.scatter_(1, data_a, 1)
    output_dims = data.size() + (vector_len,)
    return output.view(output_dims)


def safe_id_to_one_hot(data: torch.Tensor, vector_len: int, dtype: Optional[torch.dtype] = None):
    """Converts ID to one-hot representation - safe version - input can contain invalid values.

    Each element in `data` is converted into a one-hot-representation - a vector of
    length vector_len having all zeros and one 1 on the position of value of the element.

    Args:
        data: ID of a class, for IDs out of range 0 <= ID < vector_len is returned zero vector
        vector_len: length of the output tensor for each one-hot-representation
        dtype: data type of the output tensor

    Returns:
        Tensor of size [data.shape[0], vector_len] with one-hot encoding.
        For example, it converts the integer cluster indices of size [flock_size, batch_size] into
        one hot representation [flock_size, batch_size, n_cluster_centers].

    See Also:
        id_to_one_hot()
    """
    mask = (data < 0) + (data >= vector_len)
    # Mask data so scatter won't crash
    data[mask] = 0

    output = id_to_one_hot(data, vector_len, dtype)

    # Mask output so invalid ids result in zero vector
    output[mask] = 0
    return output


def one_hot_to_id(one_hot: torch.Tensor) -> torch.Tensor:
    return torch.argmax(one_hot, 1)


def clamp_tensor(input_tensor: torch.Tensor, min: Optional[torch.Tensor] = None, max: Optional[torch.Tensor] = None) ->\
        torch.Tensor:
    """
    Clamps the values independently along the dim `dim` to be between the values defined in tensors `min` and `max`.
    If a boundary tensor is not specified, no clamping is applied.

    Args:
        input_tensor: input tensor to be clamped.
        min: None or tensor containing min values, it is broadcasted over the clamped tensor.
        max: None or tensor containing max values, it is broadcasted over the clamped tensor.

    Returns:

    """
    if max is not None:
        clamped_max = torch.min(input_tensor, max)
    else:
        clamped_max = input_tensor

    if min is not None:
        clamped_min = torch.max(clamped_max, min)
    else:
        clamped_min = clamped_max

    return clamped_min


def memory_profiling(func):
    global delta_alloc_dict
    global delta_max_cached_dict
    global call_dict
    global into_tensor_alloc
    global into_cached_alloc
    global subcall_tensor_allocs
    global subcall_cache_allocs
    global ncalls_dict

    delta_max_cached_dict = {}
    delta_alloc_dict = {}
    call_dict = {}
    into_tensor_alloc = {}
    into_cached_alloc = {}
    subcall_tensor_allocs = {}
    subcall_cache_allocs = {}
    ncalls_dict = {}

    sys.settrace(trace_calls)
    func()
    sys.settrace(None)

    write_stats()


def write_stats():
    t = PrettyTable(["Name", "n_calls", "Cache + (MB)", "Cache percall", "Max cache", "Memory +", "Memory -", "Mem + percall", "Max memory +"])
    t.float_format = "4.2"
    for name in ncalls_dict.keys():
        try:

            cache_alloc_dealloc = np.array(delta_max_cached_dict[name])
            cache_alloc = sum(cache_alloc_dealloc[cache_alloc_dealloc > 0])
            cache_avg = cache_alloc / ncalls_dict[name]
            if len(cache_alloc_dealloc[cache_alloc_dealloc > 0]) > 0:
                max_cache = max(cache_alloc_dealloc[cache_alloc_dealloc > 0])
            else:
                max_cache = 0

            mem_alloc_dealloc = np.array(delta_alloc_dict[name])
            mem_alloc = sum(mem_alloc_dealloc)
            if mem_alloc < 0:
                mem_alloc = 0

            mem_dealloc = sum(mem_alloc_dealloc)
            if mem_alloc > 0:
                mem_dealloc = 0

            alloc_avg = mem_alloc / ncalls_dict[name]

            t.add_row([name, ncalls_dict[name], cache_alloc/(1024 ** 2), cache_avg/(1024 ** 2),
                       max_cache/(1024 ** 2), mem_alloc/(1024 ** 2), mem_dealloc/(1024 ** 2), alloc_avg/(1024 ** 2),
                       max(delta_alloc_dict[name])/(1024 ** 2)])
        except:
            print(f"Function {name}: No info")

    print("Memory trace, ordered by 'Memory +' (How much memory this function allocates, excluding subcalls)")
    print(t.get_string(sortby="Memory +", reversesort=True))


def trace_inside(frame, event, arg):
    if event != "return":
        return
    # This function has finished
    # Get its name
    func_name = get_name(frame)

    # Get its caller
    caller = frame.f_back
    caller_name = get_name(caller)

    # Get the differences in cache and tensor allocations, removing any allocations
    out_tensor_alloc = torch.cuda.memory_allocated()
    out_cached = torch.cuda.max_memory_cached()

    mem_change_this_call = out_tensor_alloc - into_tensor_alloc[func_name]
    alloc_this_call = mem_change_this_call - subcall_tensor_allocs[func_name]

    cache_change_this_call = out_cached - into_cached_alloc[func_name]
    cached_this_call = cache_change_this_call - subcall_cache_allocs[func_name]

    delta_alloc_dict[func_name].append(alloc_this_call)
    delta_max_cached_dict[func_name].append(cached_this_call)

    # All the memory allocated this call should be added
    subcall_tensor_allocs[caller_name] += mem_change_this_call
    subcall_cache_allocs[caller_name] += cache_change_this_call


def trace_calls(frame, event, arg):
    if event != 'call':
        return

    func_name = get_name(frame)

    caller = frame.f_back
    if caller is None:
        return
    else:
        caller_name = get_name(caller)
    #print(f"Call to {func_name} from {caller_name}")

    # Set up stats if needed
    if func_name not in delta_alloc_dict:
        delta_alloc_dict[func_name] = []
    if func_name not in delta_max_cached_dict:
        delta_max_cached_dict[func_name] = []
    if func_name not in subcall_tensor_allocs:
        subcall_tensor_allocs[func_name] = 0
    if func_name not in subcall_cache_allocs:
        subcall_cache_allocs[func_name] = 0
    if func_name not in ncalls_dict:
        ncalls_dict[func_name] = 0

    if caller_name not in call_dict:
        call_dict[caller_name] = set()
    if caller_name not in subcall_tensor_allocs:
        subcall_tensor_allocs[caller_name] = 0
    if caller_name not in subcall_cache_allocs:
        subcall_cache_allocs[caller_name] = 0

    # Record that this function has been called at least once by the caller
    call_dict[caller_name].add(func_name)

    # Set the initial conditions for memory profiling this function
    into_tensor_alloc[func_name] = torch.cuda.memory_allocated()
    into_cached_alloc[func_name] = torch.cuda.max_memory_cached()

    subcall_tensor_allocs[func_name] = 0
    subcall_cache_allocs[func_name] = 0

    # Increment the call counter
    ncalls_dict[func_name] += 1

    return trace_inside


def get_name(frame):

    co = frame.f_code
    func_name = co.co_name
    filename = co.co_filename.split("/")[-2:]

    return f"{filename[0]}/{filename[1]}:{func_name}"