from typing import List

import numpy as np
import pytest
import torch

from torchsim.core.datasets.mnist import LimitedDatasetMNIST
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.dataset_mnist_node import DatasetMNISTParams, DatasetMNISTNode, DatasetSequenceMNISTNodeParams
from torchsim.core.eval.node_accessors.mnist_node_accessor import MnistNodeAccessor
from torchsim.core.nodes.dataset_sequence_mnist_node import DatasetSequenceMNISTNode
from torchsim.core.utils.tensor_utils import same

DATASET_LIMIT = 100


def get_dataset():
    return LimitedDatasetMNIST(DATASET_LIMIT)


def collect_labels(node: DatasetMNISTNode, n_samples: int) -> List[int]:
    labels = []

    for sample in range(0, n_samples):
        node.step()
        labels.append(MnistNodeAccessor.get_label_id(node))

    return labels


def collect_data(node: DatasetMNISTNode, n_samples: int) -> List[int]:
    labels = []
    bitmaps = []

    for sample in range(0, n_samples):
        node.step()
        labels.append(MnistNodeAccessor.get_label_id(node))
        bitmaps.append(MnistNodeAccessor.get_data(node))

    return labels, bitmaps


def sequences_equal(list_a: List[int], list_b: List[int]) -> bool:
    if len(list_a) != len(list_b):
        return False

    for pos in range(0, len(list_a)):
        if list_a[pos] != list_b[pos]:
            return False

    return True


@pytest.mark.slow
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('seeds', [(None, None), (1, 1), (1, 104)])
def test_mnist_node_determinism(device, seeds):
    """Test expected (non-)determinism of MNISTNode.

    I.e. that:
        -two MNISTNodes with None seeds produce unique data,
        -two MNISTNodes with equal seeds produce equal data
        -and this is not influenced by changing some other random generators

    If seed is:
     None - generate unique sequences
     different - generate unique sequences
     equal - generate identical sequences

    Should work on CPU/GPU
    """

    # common params
    params = DatasetMNISTParams()
    params.one_hot_labels = False
    params.random_order = True

    n_samples = 100
    creator = AllocatingCreator(device=device)

    node = DatasetMNISTNode(params, dataset=get_dataset(), seed=seeds[0])
    do_some_ugly_cpu_random_stuff()
    node.allocate_memory_blocks(creator)
    do_some_ugly_cpu_random_stuff()
    labels_a = collect_labels(node, n_samples)

    node = DatasetMNISTNode(params, dataset=get_dataset(), seed=seeds[1])
    do_some_ugly_cpu_random_stuff()
    node.allocate_memory_blocks(creator)
    do_some_ugly_cpu_random_stuff()
    labels_b = collect_labels(node, n_samples)

    equal = sequences_equal(labels_a, labels_b)

    if seeds[0] is None and seeds[1] is None:
        assert not equal
    elif seeds[0] == seeds[1]:
        assert equal
    else:
        assert not equal


def collect_allowed_labels(seq_params: DatasetSequenceMNISTNodeParams) -> List[int]:
    """Collect all labels from the DatasetSequenceMNISTNodeParams and return them as a list."""
    all_labels = []
    sequences: List[List[int]] = seq_params.seqs

    for seq in sequences:
        all_labels = list(set().union(all_labels, seq))

    return all_labels


def labels_contained_in_params(labels: List[int], seq_params: DatasetSequenceMNISTNodeParams) -> bool:
    """Is everything in the list of labels contained in the sequence params?"""
    allowed_labels = collect_allowed_labels(seq_params)

    for label in labels:
        if label not in allowed_labels:
            return False

    return True


def test_allowed_labels():
    """Test the method used for testing."""
    seq_params = DatasetSequenceMNISTNodeParams([[1, 2, 9]])
    all = collect_allowed_labels(seq_params)

    assert len(all) == 3
    assert all[0] == 1
    assert all[1] == 2
    assert all[2] == 9


def test_allowed_labels_more_sequences():

    seq_params = DatasetSequenceMNISTNodeParams([[1, 2, 9], [9], [1, 2, 3, 4]])
    all = collect_allowed_labels(seq_params)

    assert len(all) == 5
    assert 1 in all
    assert 2 in all
    assert 9 in all
    assert 3 in all
    assert 4 in all


def do_some_ugly_cpu_random_stuff():
    r1 = np.random.RandomState()
    r1.seed(seed=1234)
    r1.randint(123, size=100)

    np.random.seed(54321)
    np.random.sample()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('seed', [None, 10])
def test_sequence_generator_deterministic_nondeterministic(device, seed):
    """Sequence generator has to generate repeatable sequences with custom seed, unique ones with None seed.

    Has to work on GPU/CPU.
    """
    creator = AllocatingCreator(device=device)

    params = DatasetMNISTParams()

    # uniform prob transition probabilities
    seq_params = DatasetSequenceMNISTNodeParams([[1, 2, 3], [9, 8, 7], [2, 3], [7]])

    node = DatasetSequenceMNISTNode(params, seq_params=seq_params, dataset=get_dataset(), seed=seed)
    node.allocate_memory_blocks(creator)

    n_steps = 100

    labels = collect_labels(node, n_steps)

    assert labels_contained_in_params(labels, seq_params)

    # create another node with the same params & seed
    node = DatasetSequenceMNISTNode(params, seq_params=seq_params, dataset=get_dataset(), seed=seed)
    do_some_ugly_cpu_random_stuff()     # do strange things before the unit created
    node.allocate_memory_blocks(creator)
    do_some_ugly_cpu_random_stuff()     # do strange things after the unit is created
    labels_b = collect_labels(node, n_steps)

    equal = sequences_equal(labels, labels_b)

    if seed is None:
        assert not equal
    else:
        assert equal


def labels_in_filter(labels: List[int], filter: List[int]) -> bool:
    for label in labels:
        if label not in filter:
            return False

    return True


@pytest.mark.parametrize('examples_per_class', [None, 1, 11])
def test_class_filter(examples_per_class: int):
    """Should be able to generate from the restricted set of class labels."""

    params = DatasetMNISTParams()
    params.class_filter = [0, 1, 2, 7, 8, 9]
    params.one_hot_labels = False
    params.examples_per_class = None

    n_samples = 100

    node = DatasetMNISTNode(params, dataset=get_dataset(), seed=None)
    node.allocate_memory_blocks(AllocatingCreator('cpu'))
    labels = collect_labels(node, n_samples)

    assert labels_in_filter(labels, params.class_filter)


def add_tensor_if_not_there(known_bitmaps: List[torch.Tensor], bitmap: torch.Tensor):
    """Go through the list of known bitmaps, add if not there."""
    for known_bitmap in known_bitmaps:
        if same(known_bitmap, bitmap):
            return

    known_bitmaps.append(bitmap)


def put_tensors_in_dictionary(labels: List[int], bitmaps: List[torch.Tensor], class_filter: List[int]):
    dictionary = dict()
    for value in class_filter:
        dictionary[value] = []

    # collect unique bitmaps, each for own list according to the class
    for pos in range(0, len(labels)):
        # print(f'label {labels[pos]} x {bitmaps[pos]}')
        add_tensor_if_not_there(dictionary[labels[pos]], bitmaps[pos])

    return dictionary


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize('generator_sequences', [None, [[1, 2, 3], [5, 2]]])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
@pytest.mark.parametrize('random_order', [True, False])
def test_examples_per_class(device: str, generator_sequences, random_order: bool):
    """Limit the examples_per_class.

    Should work:
        with the sequence generator
        without it with sequential loading of the bitmaps
        without it with random loading of the bitmaps
        on CPU/GPU
    """

    if generator_sequences is not None:
        generator_params = DatasetSequenceMNISTNodeParams(generator_sequences)
    else:
        generator_params = None

    params = DatasetMNISTParams()
    params.class_filter = [2, 3, 5, 7]
    params.examples_per_class = 2
    params.random_order = random_order

    n_samples = 50  # do not decrease, otherwise the test may fail due to randomness

    node = DatasetSequenceMNISTNode(params=params, seq_params=generator_params, dataset=get_dataset(), seed=None)
    node.allocate_memory_blocks(AllocatingCreator(device=device))

    # if the sequence generator is specified, it should update the class filter accordingly
    if generator_params is not None:
        assert 1 in node._unit._params.class_filter
        assert 7 not in node._unit._params.class_filter

    labels, bitmaps = collect_data(node, n_samples)

    # collect the dictionary class->List[bitmaps], check that for each class there are correct no of unique bitmaps
    dictionary = put_tensors_in_dictionary(labels, bitmaps, node._unit._params.class_filter)

    # for each class that is in the sequence_params, it should sample both (two) (unique) bitmaps
    for current_class, bitmaps in dictionary.items():
        # print(f'len bitmaps is {len(bitmaps)}')
        assert len(bitmaps) == params.examples_per_class


@pytest.mark.parametrize('device', ['cpu'])
@pytest.mark.parametrize('sequence_type', ['ordered', 'random', 'sequence_generator'])
@pytest.mark.parametrize('class_filter', [None, [1, 9, 4], [-2], [11, 2]])
def test_deterministic_samples_per_class_choice(device: str, sequence_type: str, class_filter):
    """Test random/ordered/sequential sampling of the dataset with and without class filter.

    Everything should produce valid and repeatable results if seed is not None.
    """

    params = DatasetMNISTParams()
    generator_params = None

    if sequence_type == 'ordered':
        params.random_order = False
    elif sequence_type == 'random':
        params.random_order = True
    else:
        generator_params = DatasetSequenceMNISTNodeParams([[1, 2, 3], [5, 2]])

    # parametrized, try filtered and non-filtered versions
    params.class_filter = class_filter

    params.examples_per_class = 10

    n_samples = 15

    creator = AllocatingCreator(device=device)

    node = DatasetSequenceMNISTNode(params, seq_params=generator_params, dataset=get_dataset(), seed=10)
    node.allocate_memory_blocks(creator)
    labels, bitmaps = collect_data(node, n_samples)

    node = DatasetSequenceMNISTNode(params, seq_params=generator_params, dataset=get_dataset(), seed=10)
    node.allocate_memory_blocks(creator)
    labels_b, bitmaps_b = collect_data(node, n_samples)

    assert len(bitmaps_b) == len(bitmaps)

    # go through the generated sequences and compare labels and bitmaps, should be identical
    for bitmap_a, label_a, bitmap_b, label_b in zip(bitmaps, labels, bitmaps_b, labels_b):
        # the same sequences of labels
        assert label_a == label_b
        # bitmaps are identical (randomly sampled the same samples from the dataset)
        assert same(bitmap_a, bitmap_b)


def test_sequence_mnist_node_works():
    """Just run the SequenceMNISTNode to check nothing is broken."""

    params = DatasetMNISTParams()
    generator_params = DatasetSequenceMNISTNodeParams([[1, 2, 3], [5, 2]])

    node = DatasetSequenceMNISTNode(params, seq_params=generator_params, dataset=get_dataset())
    node.allocate_memory_blocks(AllocatingCreator('cpu'))

    node.step()
    label = MnistNodeAccessor.get_label_id(node)
    bitmap = MnistNodeAccessor.get_data(node)

    assert 0 <= label < 10
    assert bitmap.shape[0] == 28 and bitmap.shape[1] == 28 and len(bitmap.shape) == 2


def test_seed_independence():
    """Create two np.random generator instances, test their independence."""
    r1 = np.random.RandomState()

    # 2x seed and sample the same
    r1.seed(seed=10)
    a1 = r1.randint(10, size=100)
    r1.seed(seed=10)
    a11 = r1.randint(10, size=100)

    # the same seed, init different generator, and sample from it
    r1.seed(seed=10)
    r2 = np.random.RandomState()
    r2.seed(seed=1234)
    a2 = r2.randint(10, size=100)

    # use the first generator without reseeding
    a111 = r1.randint(10, size=100)
    alll_no = r1.randint(10, size=100)

    # compare the sequences
    assert (a1 == a11).all()
    assert (a11 == a111).all()  # this means the first generator is not influenced by the second
    assert not (all == alll_no).all()




