import logging
import os
from dataclasses import dataclass

import matplotlib.image as mpimg
import numpy as np

import torch
from torchsim.core import FLOAT_NAN, SHARED_DRIVE
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver
from torchsim.core.test_optimizations import small_dataset_for_tests_allowed
from torchsim.gui.observer_system import ObserverPropertiesItem
from torchsim.gui.validators import *
from torchsim.utils.seed_utils import get_rand_generator_or_set_cuda_seed

logger = logging.getLogger(__name__)


class DatasetLoader:

    SHARED_DRIVE_FOLDER = os.path.join(SHARED_DRIVE, 'Datasets', 'ImageDatasets')
    LOCAL_FOLDER = os.path.join('data', 'datasets', 'image_datasets')

    _directory: str

    def __init__(self, dataset_path: str, load_snippet: bool = False):
        """Loading and initialization of dataset.
        """

        self.load_snippet = load_snippet

        if small_dataset_for_tests_allowed():
            self.load_snippet = True

        # TODO: add downloading from shared drive
        # TODO: use local addressing (just by name?)
        # self._directory = os.path.join(self.LOCAL_FOLDER, dataset_name)
        self._directory = os.path.join(dataset_path)

    @staticmethod
    def _load_dataset(directory: str, limit_to_n: Optional[int] = None) -> torch.Tensor:
        files = os.listdir(directory)

        if limit_to_n is not None:
            limit = min(limit_to_n, len(files))
            files = files[:limit]

        images = []

        for file in files:
            img = mpimg.imread(os.path.join(directory, file))
            img_np = np.asarray(img, dtype=np.float32)[:, :, :3]  # Strip alpha channel (the 4. channel) when present.

            # normalize to [0, 1]
            extension = os.path.splitext(file)[1]
            if extension == '.bmp' or extension == '.jpg':
                img_np /= 255

            images.append(img_np)

        return torch.from_numpy(np.stack(images))

    def load_dataset(self) -> torch.Tensor:
        if self.load_snippet:
            limit = 5
        else:
            limit = None

        return self._load_dataset(self._directory, limit_to_n=limit)


@dataclass
class ImagesDatasetParams(ParamsBase):
    """Class used for configuring the dataset."""

    images_path: str = os.path.join('tests', 'data', 'datasets', 'testing_dataset')  # path to the directory containing images
    random_order: bool = False  # present the samples in random order
    save_gpu_memory: bool = False  # if True, images are kept on CPU and copied to GPU only when needed


class ImagesDatasetUnit(Unit):
    """Loads a dataset with images from a directory. Then presents each image sequentially or in a random order.
    """

    _params: ImagesDatasetParams
    _images: torch.Tensor  # images loaded from a file
    last_image: torch.Tensor
    _random: np.random
    _n_samples: int

    def __init__(self, creator: TensorCreator, params: ImagesDatasetParams, random):
        super().__init__(creator.device)
        self._params = params
        self._random = random
        self._save_memory = params.save_gpu_memory

        self._images = DatasetLoader(self._params.images_path).load_dataset()
        self._n_samples = self._images.shape[0]

        if self._save_memory:
            device = 'cpu'
        else:
            device = self._device
        self._images = self._images.to(device)

        # prepare the tensors for the first step
        img_size = self._images[0].shape

        self.last_image = self._create_tensor(img_size, FLOAT_NAN, creator)

        self._current_image_id = -1

    def _create_tensor(self, shape: Union[List[int], torch.Size], fill_value: float, creator: TensorCreator):
        return creator.full(shape, fill_value=fill_value, dtype=self._float_dtype, device=self._device)

    def step(self):
        """Each step read an image (either sequentially or randomly) and put it on the output."""

        if self._params.random_order:
            self._current_image_id = self._random.randint(low=0, high=self._n_samples)
        else:  # sequential order
            self._current_image_id = (self._current_image_id + 1) % self._n_samples

        self.last_image.copy_(self._images[self._current_image_id])

    def save(self, saver: Saver):
        super()._save(saver)

        saver.description['current_image_id'] = self._current_image_id
        random_state = list(self._random.get_state())
        random_state[1] = [int(value) for value in random_state[1]]
        saver.description['_random_state'] = tuple(random_state)

    def load(self, loader: Loader):
        super()._load(loader)

        self._current_image_id = loader.description['current_image_id']
        self._random.set_state(loader.description['_random_state'])


class ImagesDatasetOutputs(MemoryBlocks):
    """image_output: width by height RGB image.

    """
    def __init__(self, owner):
        super().__init__(owner)
        self.output_image = self.create("Output_image")

    def prepare_slots(self, unit: ImagesDatasetUnit):
        self.output_image.tensor = unit.last_image


class ImagesDatasetNode(WorkerNodeBase[EmptyInputs, ImagesDatasetOutputs]):
    """loads and presents images from a dataset in a sequential or random order.
    """

    _unit: ImagesDatasetUnit
    _seed: int
    _params: ImagesDatasetParams

    def __init__(self, params: ImagesDatasetParams, seed: int = None, name: str = "ImagesDataset"):
        super().__init__(name=name,
                         outputs=ImagesDatasetOutputs(self),
                         inputs=EmptyInputs(self))
        self._seed = seed
        self._params = params.clone()

    def _create_unit(self, creator: TensorCreator):
        random = get_rand_generator_or_set_cuda_seed(creator.device, self._seed)

        return ImagesDatasetUnit(creator, self._params, random)

    @property
    def seed(self)-> int:
        return self._seed

    @seed.setter
    def seed(self, value: int):
        validate_positive_int(value)
        self._seed = value

    @property
    def images_path(self) -> str:
        return self._params.images_path

    @images_path.setter
    def images_path(self, value: str):
        self._params.images_path = value

    @property
    def random_order(self) -> bool:
        return self._params.random_order

    @random_order.setter
    def random_order(self, value: bool):
        self._params.random_order = value

    @property
    def save_gpu_memory(self) -> bool:
        return self._params.save_gpu_memory

    @save_gpu_memory.setter
    def save_gpu_memory(self, value: bool):
        self._params.save_gpu_memory = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        """Define which properties can be changed from GUI and how."""
        return [
            self._prop_builder.auto('Seed', type(self).seed),
            self._prop_builder.auto('Path to the dataset', type(self).images_path),
            self._prop_builder.auto('Random order', type(self).random_order),
            self._prop_builder.auto('Save gpu memory', type(self).save_gpu_memory),
        ]

    def _step(self):
        self._unit.step()

    def validate(self):
        pass


