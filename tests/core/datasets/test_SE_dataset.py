import pytest
import torch

from torchsim.core import get_float
from torchsim.core.datasets.space_divisor import SpaceDivisor
from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.datasets.dataset_se_task_zero import DatasetSeTask0
from torchsim.core.datasets.dataset_se_task_one import DatasetSeTask1
from torchsim.core.utils.tensor_utils import same
from pytest import raises


CONFIG_LABEL_SIZE = 20


@pytest.mark.parametrize('dataset_size', [SeDatasetSize.SIZE_24, SeDatasetSize.SIZE_32, SeDatasetSize.SIZE_64])
def test_t0_available_sizes(dataset_size: SeDatasetSize):

    header, train, test = DatasetSeTask0(dataset_size, load_snippet=False).get_all()
    train_data = train[0]
    train_labels = train[1]
    train_instance_id = train[2]
    train_examples_per_class = train[3]

    test_data = test[0]
    test_labels = test[1]
    test_instance_id = test[2]
    test_examples_per_class = test[3]

    assert train_data.shape[1] == dataset_size.value
    assert train_data.shape[2] == dataset_size.value
    assert train_data.shape[3] == DatasetSeTask0.N_CHANNELS
    assert train_data.shape[0] == train_labels.shape[0]
    assert train_labels.shape[1] == CONFIG_LABEL_SIZE
    assert train_instance_id.shape[0] == train_labels.shape[0]
    assert train_examples_per_class.shape[0] > 0
    assert test_data.shape[1] == dataset_size.value
    assert test_data.shape[2] == dataset_size.value
    assert test_data.shape[3] == DatasetSeTask0.N_CHANNELS
    assert test_data.shape[0] == test_labels.shape[0]
    assert test_labels.shape[1] == CONFIG_LABEL_SIZE
    assert test_instance_id.shape[0] == test_labels.shape[0]
    assert test_examples_per_class.shape[0] > 0


def test_t0_unavailable_sizes():
    with raises(Exception):
        DatasetSeTask0(SeDatasetSize.SIZE_128)
    with raises(Exception):
        DatasetSeTask0(SeDatasetSize.SIZE_256)


def test_t1_24():
    header, train_data, train_labels = DatasetSeTask1(SeDatasetSize.SIZE_24, load_snippet=False).get_all()
    assert train_data.shape[1] == SeDatasetSize.SIZE_24.value
    assert train_data.shape[2] == SeDatasetSize.SIZE_24.value
    assert train_data.shape[3] == DatasetSeTask1.N_CHANNELS
    assert train_data.shape[0] == train_labels.shape[0]
    assert train_labels.shape[1] == 2


def test_t1_32():
    header, train_data, train_labels = DatasetSeTask1(SeDatasetSize.SIZE_32, load_snippet=True).get_all()
    assert train_data.shape[1] == SeDatasetSize.SIZE_32.value
    assert train_data.shape[2] == SeDatasetSize.SIZE_32.value
    assert train_data.shape[3] == DatasetSeTask1.N_CHANNELS
    assert train_data.shape[0] == train_labels.shape[0]
    assert train_labels.shape[1] == 2


@pytest.mark.slow
def test_t1_64():
    header, train_data, train_labels = DatasetSeTask1(SeDatasetSize.SIZE_64, load_snippet=True).get_all()
    assert train_data.shape[1] == SeDatasetSize.SIZE_64.value
    assert train_data.shape[2] == SeDatasetSize.SIZE_64.value
    assert train_data.shape[3] == DatasetSeTask1.N_CHANNELS
    assert train_data.shape[0] == train_labels.shape[0]
    assert train_labels.shape[1] == 2


@pytest.mark.slow
def test_t1_128():
    header, train_data, train_labels = DatasetSeTask1(SeDatasetSize.SIZE_128, load_snippet=True).get_all()
    assert train_data.shape[1] == SeDatasetSize.SIZE_128.value
    assert train_data.shape[2] == SeDatasetSize.SIZE_128.value
    assert train_data.shape[3] == DatasetSeTask1.N_CHANNELS
    assert train_data.shape[0] == train_labels.shape[0]
    assert train_labels.shape[1] == 2


@pytest.mark.slow
def test_t1_256():
    header, train_data, train_labels = DatasetSeTask1(SeDatasetSize.SIZE_256, load_snippet=True).get_all()
    assert train_data.shape[1] == SeDatasetSize.SIZE_256.value
    assert train_data.shape[2] == SeDatasetSize.SIZE_256.value
    assert train_data.shape[3] == DatasetSeTask1.N_CHANNELS
    assert train_data.shape[0] == train_labels.shape[0]
    assert train_labels.shape[1] == 2


def get_pos(pos: int, source: torch.Tensor) -> [float, float]:
    return [source[pos, 0].item(), source[pos, 1].item()]


def get_data(device: str):
    """Define test data for the landmarks."""

    positions = torch.tensor([0, 0,
                              40, 0,
                              70, 0,
                              99, 0,
                              0, 55,
                              30, 55,
                              60, 55,
                              99.9, 55,
                              0, 99,
                              30, 99,
                              60, 98,
                              97, 98], device=device, dtype=get_float(device) ).view(-1, 2)/100

    results = torch.tensor([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], device=device, dtype=get_float(device))
    return positions, results


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_landmarks(device: str):
    """Test whether the get_landmark and get_landmarks work consistently and correctly."""

    divisor = SpaceDivisor(horizontal_segments=3,
                           vertical_segments=4,
                           device=device)
    # test data
    positions, results = get_data(device)
    no_positions = positions.shape[0]
    assert no_positions == results.numel()

    # get landmarks for the entire test data
    landmarks, _ = divisor.get_landmarks(positions)
    assert landmarks.shape[0] == no_positions

    # sequential get landmark for one position, compare with:
    #   -expected landmark and
    #   -with landmark computed by the other method
    for position in range(0, no_positions):
        coordinates = get_pos(position, positions)

        assert divisor.get_landmark(*coordinates)[0] == results[position]
        assert landmarks[position] == results[position]


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_one_hot(device: str):
    """Test whether the one-hot representation works correctly as well."""

    # request one-hot representation of landmarks
    one_hot_divisor = SpaceDivisor(horizontal_segments=3, vertical_segments=4, device=device)

    positions, results = get_data(device)
    no_positions = positions.shape[0]

    _, one_hot_landmarks = one_hot_divisor.get_landmarks(positions)

    for position in range(0, no_positions):
        coordinates = get_pos(position, positions)

        _, one_hot_l = one_hot_divisor.get_landmark(*coordinates)

        assert same(one_hot_l, one_hot_landmarks[position])

        val, index = one_hot_l.max(-1)

        assert index == results[position].long()