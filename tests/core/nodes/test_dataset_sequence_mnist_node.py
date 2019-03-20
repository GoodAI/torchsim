import pytest
import torch

from torchsim.core import get_float
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.dataset_mnist_node import DatasetMNISTParams, DatasetSequenceMNISTNodeParams
from torchsim.core.nodes.dataset_sequence_mnist_node import DatasetSequenceMNISTNode
from torchsim.core.utils.tensor_utils import same, normalize_probs


@pytest.mark.slow
@pytest.mark.parametrize("expected_sequence, seq_params", [(torch.tensor([0, 1, 2, 3]), DatasetSequenceMNISTNodeParams([[0, 1, 2, 3]]))])
def test_sequence_iteration(expected_sequence, seq_params):
    node = DatasetSequenceMNISTNode(params=DatasetMNISTParams(one_hot_labels=False), seq_params=seq_params)
    node.allocate_memory_blocks(AllocatingCreator(device='cpu'))

    actual_sequence = []
    for x in range(len(node._seq_params.seqs[0])):
        node.step()
        actual_sequence.append(node.outputs.label.tensor.clone())

    actual_sequence = torch.cat(actual_sequence, dim=0)

    assert same(expected_sequence, actual_sequence)


@pytest.mark.parametrize("sequences, trans_probs, iters", [([[0], [1], [2], [4]], [[0.0, 1., 0.0, 0.0],
                                                                                   [0.0, 0.0, 1., 0.],
                                                                                   [0.0, 0.0, 0.0, 1.],
                                                                                   [1., 0.0, 0.0, 0.0]], 100)])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
def test_sequence_transition(sequences, trans_probs, iters, device):
    """For ease of use, sequences should be in ascending order."""

    float_dtype = get_float(device)
    seq_params = DatasetSequenceMNISTNodeParams(sequences, transition_probs=trans_probs)
    seq_params.use_custom_probs = True
    node = DatasetSequenceMNISTNode(params=DatasetMNISTParams(one_hot_labels=False), seq_params=seq_params)
    node.allocate_memory_blocks(AllocatingCreator(device=device))

    actual_sequence = []
    for x in range(iters):
        node.step()
        actual_sequence.append(node.outputs.label.tensor.clone())

    actual_sequence = torch.cat(actual_sequence, dim=0)
    proportions = torch.bincount(actual_sequence).type(float_dtype) / iters
    real_proportions = proportions[proportions > 0]

    expected_proportions = normalize_probs(torch.tensor(trans_probs, dtype=float_dtype, device=device ).sum(0), dim=0)

    assert same(expected_proportions, real_proportions, eps=1e-2)
