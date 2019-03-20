import os
from typing import NamedTuple

import torch

from torchsim.core import SHARED_DRIVE
from torchsim.core.datasets.dataset_se_base import DatasetSeBase, SeDatasetSize
from torchsim.core.test_optimizations import small_dataset_for_tests_allowed


class DatasetResult(NamedTuple):
    data: torch.Tensor
    labels: torch.Tensor


class DatasetSeTask0(DatasetSeBase):

    NAME_PREFIX = 'SE_T0'

    NAME_POSTFIX_TRAIN = '_train'
    NAME_POSTFIX_TEST = '_test'
    SHARED_DRIVE_FOLDER = os.path.join(SHARED_DRIVE, 'Datasets', 'SE', 'objects_dataset')

    _train_filename_with_path: str
    _test_filename_with_path: str

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

        if size.value > 64:
            raise Exception("Sizes above 64x64 px are not supported!")

        train_filename = self._compose_filename(DatasetSeTask0.NAME_POSTFIX_TRAIN, load_snippet)
        test_filename = self._compose_filename(DatasetSeTask0.NAME_POSTFIX_TEST, load_snippet)

        self._train_filename_with_path = os.path.join(self.PATH_TO_DATASET, train_filename)
        self._download_if_not_found(self._train_filename_with_path, train_filename, self.SHARED_DRIVE_FOLDER)

        self._test_filename_with_path = os.path.join(self.PATH_TO_DATASET, test_filename)
        self._download_if_not_found(self._test_filename_with_path, test_filename, self.SHARED_DRIVE_FOLDER)

    def _compose_filename(self, name_postfix: str, snippet: bool):

        return DatasetSeTask0.NAME_PREFIX + \
               self._dataset_dims_to_str(self._size.value) + \
               name_postfix + \
               ('_snippet' if snippet else '_full') +\
               '.set'

    def load_dataset(self):
        header = self._dataset_dims_to_str(self._size.value)

        dataset = torch.load(self._train_filename_with_path)
        train_data = dataset[0]
        train_labels = dataset[1]
        train_instance_id = dataset[2]
        train_examples_per_class = dataset[3]

        dataset = torch.load(self._test_filename_with_path)
        test_data = dataset[0]
        test_labels = dataset[1]
        test_instance_id = dataset[2]
        test_examples_per_class = dataset[3]

        train = [train_data, train_labels, train_instance_id, train_examples_per_class]
        test = [test_data, test_labels, test_instance_id, test_examples_per_class]

        return header, train, test
