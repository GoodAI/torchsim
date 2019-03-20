from typing import List, Tuple

import torch
from itertools import chain

from torchsim.core.utils.tensor_utils import same


def randomize_subflock(should_update, should_not_update):
    for flock_tensor, subflock_tensor in chain(should_update, should_not_update):
        subflock_tensor.random_()


def calculate_expected_results(should_update, should_not_update, flock_size, indices_np):
    expected_results_updated = [[] for _ in should_update]
    expected_results_not_updated = [[] for _ in should_not_update]

    subflock_ptr = 0
    for k in range(flock_size):
        for i, (original, subflock) in enumerate(should_update):
            if k in indices_np:
                # The expert is in the subflock.
                expected_results_updated[i].append(subflock[subflock_ptr])
            else:
                # The expert is not in the subflock.
                expected_results_updated[i].append(original[k])

        if k in indices_np:
            subflock_ptr += 1

        for i, (original, subflock) in enumerate(should_not_update):
            expected_results_not_updated[i].append(original[k])

    return [torch.stack(expected) for expected in expected_results_updated + expected_results_not_updated]


def check_integration_results(
        expected_results: List[torch.Tensor],
        should_update: List[Tuple[torch.Tensor, torch.Tensor]],
        should_not_update: List[Tuple[torch.Tensor, torch.Tensor]]):
    flock_tensors = map(lambda x: x[0], chain(should_update, should_not_update))
    for expected, original in zip(expected_results, flock_tensors):
        assert same(expected, original)
