import pytest

from torchsim.core.eval.node_accessors.random_number_accessor import RandomNumberNodeAccessor
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.random_number_node import RandomNumberNode


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_rnd_node_accessor_return_type(device):

    lower_bound = 50
    upper_bound = 100

    node = RandomNumberNode(lower_bound=lower_bound, upper_bound=upper_bound)
    node.allocate_memory_blocks(AllocatingCreator(device=device))
    node._step()

    random_number = RandomNumberNodeAccessor.get_output_id(node)

    assert type(random_number) is int
    assert lower_bound <= random_number < upper_bound

