import torch
from typing import NamedTuple, List

import torchvision
from torchvision.transforms import transforms

from torchsim.core.test_optimizations import small_dataset_for_tests_allowed


class DatasetResult(NamedTuple):
    data: torch.Tensor
    labels: torch.Tensor


class DatasetMNIST:

    def __init__(self):
        self._dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.1307,), (0.3081,))]))

    def get_all(self):
        if small_dataset_for_tests_allowed():
            return DatasetResult(self._dataset.train_data[:2000], self._dataset.train_labels[:2000])
        return DatasetResult(self._dataset.train_data, self._dataset.train_labels)

    def get_filtered(self, include: List[int]):
        data, labels = self.get_all()
        indices = labels.clone().apply_(lambda x: x in include).nonzero().squeeze().long()
        lm = labels.index_select(0, indices)
        dm = data.index_select(0, indices)
        return DatasetResult(dm, lm)


class LimitedDatasetMNIST(DatasetMNIST):
    """Load just a limited number of samples to save time during unit testing."""

    _sample_limit: int

    def __init__(self, sample_limit: int):
        super().__init__()

        self._sample_limit = sample_limit

    def get_all(self):
        return DatasetResult(self._dataset.train_data[0:self._sample_limit],
                             self._dataset.train_labels[0:self._sample_limit])


