import math

import torch
import warnings
from typing import List, TypeVar

T = TypeVar('T')


def flatten_list(list_of_lists: List[List[T]]) -> List[T]:
    """Merges all sub-lists of the list `list_of_lists` into one list."""
    return [item
            for sublist in list_of_lists
            for item in sublist]


def same_lists(list1: List, list2: List, eps=None):
    """Compares if two lists are same. If it fails, it tries if they are same up to the eps difference."""

    # Check sizes.
    if len(list1) != len(list2):
        return False

    if list1 == list2:
        return True

    nans1 = [math.isnan(x) for x in list1]
    nans2 = [math.isnan(x) for x in list2]

    # Check nans.
    if nans1 != nans2:
        return False

    list1_without_nans = [x for x in list1 if not math.isnan(x)]
    list2_without_nans = [x for x in list2 if not math.isnan(x)]

    if eps is None:
        return list1_without_nans == list2_without_nans
    else:
        if eps > 1:
            warnings.warn("eps is intended to be a small number in the form '1e-n'.", stacklevel=2)
        return all(x - y < eps for x, y in zip(list1_without_nans, list2_without_nans))


def dim_prod(dimensions, start_dim=0, end_dim=-1) -> int:
    if not isinstance(dimensions, torch.Tensor):
        dimensions = torch.tensor(dimensions)

    if end_dim == -1:
        end_dim = len(dimensions) - 1

    return int(torch.prod(dimensions[start_dim:end_dim+1]).item())
