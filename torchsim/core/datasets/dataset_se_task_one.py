import os
import torch
from typing import NamedTuple

from torchsim.core import SHARED_DRIVE, FLOAT_TYPE_CPU
from torchsim.core.datasets.dataset_se_base import DatasetSeBase, SeDatasetSize
from torchsim.core.test_optimizations import small_dataset_for_tests_allowed


class DatasetResult(NamedTuple):
    data: torch.Tensor
    labels: torch.Tensor


class DatasetSeTask1(DatasetSeBase):

    NAME_PREFIX = 'SE_T1'
    NAME_POSTFIX = '_train'
    SHARED_DRIVE_FOLDER = os.path.join(SHARED_DRIVE, 'Datasets', 'SE', 'navigation_dataset')

    _filename_with_path: str

    def __init__(self, size: SeDatasetSize, load_snippet: bool = False):
        """Initialization of dataset.

        Args:
           size: Choose the dataset resolution
        """
        super().__init__(size)

        self._size = size
        self.load_snippet = load_snippet

        if small_dataset_for_tests_allowed():
            load_snippet = True

        filename = self._compose_filename(DatasetSeTask1.NAME_POSTFIX, load_snippet)

        self._filename_with_path = os.path.join(self.PATH_TO_DATASET, filename)
        self._download_if_not_found(self._filename_with_path, filename, self.SHARED_DRIVE_FOLDER)

    def _compose_filename(self, name_postfix: str, snippet: bool):

        return self.NAME_PREFIX + \
               self._dataset_dims_to_str(self._size.value) + \
               name_postfix + \
               ('_snippet' if snippet else '_full') +\
               '.set'

    def load_dataset(self):
        header = self._dataset_dims_to_str(self._size.value)

        dataset = torch.load(self._filename_with_path)
        train_data = dataset[0]
        train_labels = dataset[1].type(FLOAT_TYPE_CPU)

        return header, train_data, train_labels

