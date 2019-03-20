import abc
import os
from enum import IntEnum
from shutil import copyfile

import logging

from torchsim.core.datasets.convert_dataset import DatasetConverter

logger = logging.getLogger(__name__)


class SeDatasetSize(IntEnum):
    SIZE_24 = 24
    SIZE_32 = 32
    SIZE_64 = 64
    SIZE_128 = 128
    SIZE_256 = 256


class DatasetSeBase:
    """SE SchoolMod task dataset loader.

    Contains a header with information about screenshot dimensions and a data,
    containing screenshot image data anda label data specific to each SE task.
    """

    PATH_TO_DATASET = os.path.join('data', 'eval')
    N_CHANNELS = 3

    _datasetSize: SeDatasetSize

    def __init__(self, size: SeDatasetSize):
        """Initialization of dataset.

        Args:
            size: Choose the dataset resolution
        """

        self._datasetSize = size

    def _dataset_dims_to_str(self, dims: SeDatasetSize):
        """Convert SeDatasetDims to string for the filename."""
        return '_' + str(dims) + 'x' + str(dims)

    def _download_if_not_found(self, filename_with_path: str, filename: str, sharedrive_path: str):
        try:
            dataset_file = open(filename_with_path, 'rb')
            dataset_file.close()
        except FileNotFoundError:
            logger.info(f'File {filename_with_path} not found, copying from the shared drive')
            self._download_from_shared_drive(sharedrive_path, filename, filename_with_path)

    def _convert_if_pickle(self, filename_with_path: str):
        path, filename = os.path.split(filename_with_path)
        if '.pkl' in filename:
            converter = DatasetConverter()
            converter.convert_pickles(path)

    def _download_from_shared_drive(self, sharedrive_path: str, filename: str, destination: str):
        # Create destination dirs
        destination_dir = os.path.dirname(destination)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        src = os.path.join(sharedrive_path, filename)
        try:
            copyfile(src, destination)
        except ValueError:
            logger.error(f'DatasetSE error: could not locate dataset file either locally or on the shared drive')

    @abc.abstractmethod
    def load_dataset(self):
        pass

    def get_all(self):
        return self.load_dataset()
