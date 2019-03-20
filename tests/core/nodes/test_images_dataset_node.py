import os
from abc import ABC
from typing import Generator, List, Any

import torch
from torchsim.core import get_float
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.images_dataset_node import ImagesDatasetParams, ImagesDatasetNode, DatasetLoader
from torchsim.core.utils.tensor_utils import same, gather_from_dim
from tests.core.nodes.node_unit_test_base import NodeTestBase, AnyResult


class TestImagesDatasetNode:

    @staticmethod
    def _create_expected_images_tensor(creator: AllocatingCreator):
        device = creator.device
        dtype = get_float(device)

        expected_images_tensor = creator.full([3, 2, 3, 3], dtype=dtype, device=device, fill_value=1.0)
        expected_images_tensor[0, 1, 2, :] = creator.tensor([1.0, 0.0, 0.0])
        expected_images_tensor[1, 1, 2, :] = creator.tensor([0.0, 1.0, 0.0])
        expected_images_tensor[2, 1, 2, :] = creator.tensor([0.0, 0.0, 1.0])

        return creator.cat([expected_images_tensor, expected_images_tensor])

    def test_load_dataset(self):

        images_tensor = DatasetLoader._load_dataset(os.path.join('tests', 'data', 'datasets', 'testing_dataset'))

        device = 'cpu'
        creator = AllocatingCreator(device=device)
        expected_images_tensor = self._create_expected_images_tensor(creator)

        assert same(expected_images_tensor[:2], images_tensor[:2])
        # jpeg uses lossy compression
        assert same(expected_images_tensor[2], images_tensor[2], eps=0.01)

    def test_load_dataset_shippet(self):

        images_tensor = DatasetLoader._load_dataset(
            os.path.join('tests', 'data', 'datasets', 'testing_dataset'), limit_to_n=2)

        device = 'cpu'
        creator = AllocatingCreator(device=device)
        expected_images_tensor = self._create_expected_images_tensor(creator)

        assert len(images_tensor) == 2

        assert same(expected_images_tensor[:2], images_tensor[:2])


    class ImagesDatasetNodeTestBase(NodeTestBase, ABC):
        _expected_images_tensor: torch.Tensor

        @classmethod
        def setup_class(cls, device: str = 'cuda'):
            super().setup_class()
            cls._expected_images_tensor = TestImagesDatasetNode._create_expected_images_tensor(cls._creator)
            # skip jpeg images
            cls._skip_checking = [2, 5]

        def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
            while True:
                yield []

        def _generate_expected_results(self) -> Generator[List[Any], None, None]:
            # enumerate twice over the three images in the dataset
            for idx, image in enumerate(self._expected_images_tensor):
                # jpeg images use lossy compression, we cannot compare results with absolute precision
                if idx in self._skip_checking:
                    yield [AnyResult]
                else:
                    yield [image]

        @classmethod
        def _create_params(cls) -> ImagesDatasetParams:
            params = ImagesDatasetParams()
            params.images_path = os.path.join('tests', 'data', 'datasets', 'testing_dataset')

            return params

        def _create_node(self) -> WorkerNodeBase:
            params = self._create_params()
            return ImagesDatasetNode(params, seed=3)

    class TestImagesDatasetNodeDeterministic(ImagesDatasetNodeTestBase):
        pass

    class TestImagesDatasetNodeSaveGpuMemory(ImagesDatasetNodeTestBase):
        @classmethod
        def _create_params(cls) -> ImagesDatasetParams:
            params = super()._create_params()
            params.save_gpu_memory = True

            return params

    class TestImagesDatasetNodeRandom(ImagesDatasetNodeTestBase):

        def setup_class(cls, device: str = 'cuda'):
            super().setup_class()
            device = cls._creator.device
            dtype = get_float(device)

            indices = cls._creator.tensor([2, 0, 1, 0, 0, 0], dtype=torch.long, device=device)
            cls._expected_images_tensor.copy_(gather_from_dim(cls._expected_images_tensor, indices))
            cls._skip_checking = [0]

        @classmethod
        def _create_params(cls) -> ImagesDatasetParams:
            params = super()._create_params()
            params.random_order = True

            return params







