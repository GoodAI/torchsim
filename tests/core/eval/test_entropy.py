import pytest

import torch

from torchsim.core.eval.metrics.entropy import one_hot_entropy


def test_one_hot_entropy():
    n_samples, n_components, n_elements = 10, 5, 8
    data = torch.zeros(n_samples, n_components, n_elements)
    for sample in range(n_samples):
        for component in range(n_components):
            if 0 <= component <= 1:  # entropy == 0
                data[sample, component, 0] = 1
            else:  # entropy == 1
                data[sample, component, sample % 2] = 1
    assert one_hot_entropy(data) == 3

