import torch

from torchsim.core.graph.unit import Unit
from torchsim.core.models.flock import Flock
from torchsim.core.models.flock.buffer import Buffer, BufferStorage


def dummy_printer(text: str):
    pass


def memory_report(obj, level=0, parents=None, units=None, printer=print):
    """Prints all tensors in Node, Flock or Unit and recursively in its member tensor containers with their size.

    Returns:
         The total memory used.
    """
    if parents is None:
        parents = []

    if printer is None:
        printer = dummy_printer

    prefix = "\t" * level
    printer(f"\n{prefix}Object of type {type(obj)}.")
    stats = get_tensor_stats(obj)
    subcontainers = get_subcontainers(obj)
    all_mems = []

    for name, (size, mem_size) in stats.items():
        all_mems.append(mem_size)
        rounded, suffix = round_memory(mem_size, units=units)
        name_string = name.ljust(50)
        size_string = f"{list(size)}".ljust(25)
        mem_string = f"{rounded:.2f} {suffix}"
        printer(f"{prefix}{name_string}{size_string}{mem_string}")

    parents.append(obj)
    for subobj in subcontainers:
        if subobj not in parents:
            all_mems.append(memory_report(subobj, level=level+1, parents=parents, units=units, printer=printer))

    rounded, suffix = round_memory(sum(all_mems), units=units)
    printer(f"{prefix}** Total mem used: {rounded}{suffix}.\n")
    return sum(all_mems)


def get_tensor_stats(obj):
    """Gets statistics of all tensors of the given object."""
    result = {}
    for t_name in obj.__dict__:
        var = obj.__dict__[t_name]
        if isinstance(var, torch.Tensor):
            result[t_name] = (var.size(), var.numel()*var.element_size())

    return result


def get_subcontainers(obj):
    """Gets all members which are tensor-holding containers."""
    result = []
    for t_name in obj.__dict__:
        var = obj.__dict__[t_name]
        cls = type(var)
        if issubclass(cls, Buffer) or issubclass(cls, Flock) or issubclass(cls, BufferStorage) or issubclass(cls, Unit):
            result.append(var)
    return result


def round_memory(mem, units=None):
    suffixes = ["B", "KB", "MB", "GB"]

    if units is not None and units in suffixes:
        return mem / (1024 ** suffixes.index(units)), units

    for k in reversed(range(len(suffixes))):
        if mem / (1024 ** k) > 1:
            return mem / (1024 ** k), suffixes[k]
    return mem, "B"