from typing import List

import pytest
import torch

from torchsim.core.eval.node_accessors.mnist_node_accessor import MnistNodeAccessor
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.dataset_mnist_node import DatasetMNISTParams, DatasetMNISTNode, DatasetSequenceMNISTNodeParams
from torchsim.core.nodes.dataset_sequence_mnist_node import DatasetSequenceMNISTNode
from tests.core.nodes.test_dataset_mnist_node import get_dataset, sequences_equal


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('generator', [[0, 1, 2], 0, 1])
def test_node_accessor(generator, device):
    """Validate the accessor properties."""
    # common params
    params = DatasetMNISTParams()
    params.one_hot_labels = False

    seq = None

    # different iterators might result in different behavior here
    if type(generator) is List:
        seq = DatasetSequenceMNISTNodeParams(seqs=generator)
    elif generator == 0:
        params.random_order = True
    else:
        params.random_order = False

    node = DatasetSequenceMNISTNode(params, seq_params=seq, dataset=get_dataset(), seed=123)
    node.allocate_memory_blocks(AllocatingCreator(device=device))

    node.step()
    bitmap = MnistNodeAccessor.get_data(node)
    label = MnistNodeAccessor.get_label_id(node)

    assert type(label) is int
    assert type(bitmap) is torch.Tensor
    assert sequences_equal(bitmap.shape, [28, 28])

    assert 0 <= label < 10