import pytest
import torch

from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.utils.tensor_utils import same


class DummyNode(WorkerNodeBase):
    def _create_unit(self, creator: TensorCreator):
        pass

    def _step(self):
        pass


def test_owner():
    dummy_node = DummyNode()
    tensor = torch.zeros([1,2])

    mb = MemoryBlock(dummy_node)
    mb.tensor = tensor

    assert mb.owner == dummy_node


def test_tensor_reference():
    device = 'cpu'
    node = DummyNode()

    tens = torch.zeros([10, 12]).to(device).random_()
    tens_copy = tens
    another_tensor = torch.zeros([10, 12]).to(device).random_()

    mem = MemoryBlock(node)
    mem.tensor = tens

    assert mem.tensor is tens
    assert mem.tensor is tens_copy
    assert mem.tensor is not another_tensor

    assert same(mem.tensor, tens)
    assert not same(mem.tensor, another_tensor)

    tens.random_()

    assert mem.tensor is tens
    assert mem.tensor is tens_copy
    assert same(mem.tensor, tens)


@pytest.mark.parametrize('tensor_shape, interpreted_dim, interpretation, expected_interpret_shape', [
    ([3, 20, 3, 1, 2], 1, (2, 1, 10), [3, 2, 1, 10, 3, 1, 2]),  # middle dim
    ([3, 20, 3, 1, 2], -4, (2, 1, 10), [3, 2, 1, 10, 3, 1, 2]),  # middle dim as -X
    ([1, 2, 3, 70], 3, (7, 10), [1, 2, 3, 7, 10]),  # last dim
    ([1, 2, 3, 70], -1, (7, 10), [1, 2, 3, 7, 10]),  # last dim as -1
    ([4, 3, 2], 0, (2, 2), [2, 2, 3, 2]),  # first dim
    ([4, 3, 2], -3, (2, 2), [2, 2, 3, 2]),  # first dim as -X
    ([4, 3, 2], 1, (3,), [4, 3, 2])  # just one interpreted dim
])
def test_add_interpretation(tensor_shape, interpreted_dim, interpretation, expected_interpret_shape):
    dummy_node = DummyNode()
    tensor = torch.zeros(tensor_shape)

    mb = MemoryBlock(dummy_node, 'mb')
    mb.tensor = tensor

    mb.reshape_tensor(interpretation, interpreted_dim)

    assert list(mb.shape) == expected_interpret_shape
