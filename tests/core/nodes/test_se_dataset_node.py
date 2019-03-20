from typing import List

import pytest
import torch

from torchsim.core.datasets.space_divisor import SpaceDivisor
from torchsim.core.datasets.dataset_se_base import SeDatasetSize, DatasetSeBase
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.dataset_se_navigation_node import DatasetSENavigationParams, DatasetSeNavigationNode, \
    SamplingMethod
from torchsim.core.utils.tensor_utils import same


def collect_data(node: DatasetSeNavigationNode, n_samples: int):
    labels = []
    images = []
    landmark_scalars = []
    landmarks = []

    for step in range(0, n_samples):
        node._step()
        image = node.outputs.image_output.tensor
        label = node.outputs.task_to_agent_location.tensor
        landmark_scalar = node.outputs.task_to_agent_location_int.tensor
        landmark = node.outputs.task_to_agent_location_one_hot.tensor

        images.append(image.clone())
        labels.append(label.clone())
        landmark_scalars.append(landmark_scalar.clone())
        landmarks.append(landmark.clone())

    return images, labels, landmark_scalars, landmarks


def sequences_equal(labels_a: List[torch.Tensor], images_a: List[torch.Tensor],
                    labels_b: List[torch.Tensor], images_b: List[torch.Tensor]) -> bool:

    if len(labels_a) != len(labels_b) or len(images_a) != len(images_b) or len(labels_a) != len(images_b):
        return False

    for label_a, label_b, image_a, image_b in zip(labels_a, labels_b, images_a, images_b):
        if not same(label_a, label_b):
            return False
        if not same(image_a, image_b):
            return False

    return True


class DatasetSE(object):
    pass


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('seed', [None, 10])
@pytest.mark.parametrize('sampling_method', [e for e in SamplingMethod])
def test_dataset_determinism(device, seed, sampling_method):
    """Should produce identical sequences, only if seed is None and random order is required, they should be unique."""

    n_steps = 7

    params = DatasetSENavigationParams(SeDatasetSize.SIZE_24)

    # parametrized:
    params.sampling_method = sampling_method

    node = DatasetSeNavigationNode(params, seed=seed)
    node.allocate_memory_blocks(AllocatingCreator(device))
    images, labels, _, node_landmarks = collect_data(node, n_steps)

    horizontal_segments = 3
    vertical_segments = 4

    divisor = SpaceDivisor(horizontal_segments, vertical_segments, device=device)
    landmarks = divisor.get_landmarks(torch.stack(labels))

    # correct no samples
    assert len(images) == len(labels)
    assert len(images) == n_steps

    # correct dimensions
    img = images[0]
    label = labels[0]
    landmark = landmarks[0]
    assert img.shape[2] == DatasetSeBase.N_CHANNELS
    assert img.shape[1] == SeDatasetSize.SIZE_24.value
    assert img.shape[0] == SeDatasetSize.SIZE_24.value
    assert label.shape[0] == 2
    assert len(divisor.get_landmarks(torch.stack(labels))[0].shape) == 1

    # cuda/cpu
    assert img.device.type == device
    assert label.device.type == device
    assert landmark.device.type == device

    # another run
    node = DatasetSeNavigationNode(params, seed=seed)
    node.allocate_memory_blocks(AllocatingCreator(device))
    images_b, labels_b, _, node_landmarks_b = collect_data(node, n_steps)

    # series are identical?
    equal = sequences_equal(labels, images, labels_b, images_b)

    if seed is None:
        if params.sampling_method == SamplingMethod.ORDERED:
            assert equal
        elif params.sampling_method == SamplingMethod.RANDOM_ORDER:
            assert not equal
        # SamplingMethod.RANDOM_SAMPLING cannot be tested here because it could be equal
    else:
        assert equal


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_one_hot_landmark_id(device):
    """Should produce one-hot landmark id if requested."""

    n_steps = 5

    params = DatasetSENavigationParams(dataset_size=SeDatasetSize.SIZE_24)

    # parametrized:
    params.sampling_method = SamplingMethod.RANDOM_SAMPLING
    params.horizontal_segments = 3
    params.vertical_segments = 4

    node = DatasetSeNavigationNode(params)
    node.allocate_memory_blocks(AllocatingCreator(device))
    images, labels, node_landmark_scalars, node_landmarks = collect_data(node, n_steps)

    dummy_divisor = SpaceDivisor(params.horizontal_segments,
                                 params.vertical_segments,
                                 device)

    assert node_landmarks[0].shape[0] == dummy_divisor.num_landmarks
    assert node_landmark_scalars[0].numel() == 1

