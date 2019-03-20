import pytest
import torch

from torchsim.core.eval.node_accessors.random_number_accessor import RandomNumberNodeAccessor
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.random_number_node import RandomNumberUnit, RandomNumberNode
from torchsim.utils.seed_utils import get_rand_generator_or_set_cuda_seed, set_global_seeds

from tests.core.nodes.node_unit_test_base import NodeTestBase

SEED = 2
LOWER_BOUND = 2
UPPER_BOUND = 5
STEPS = 4

# with the seed=2 and lower_bound=2, upper_bound=5 should generate 2->3->2->4
SEQUENCE = [2, 3, 2, 4]


class TestRandomNumberNode(NodeTestBase):
    input_values = [[0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1]]

    @classmethod
    def setup_class(cls, device: str = 'cuda'):
        super().setup_class(device)

    def _generate_input_tensors(self):
        for _ in SEQUENCE:
            yield []

    def _generate_expected_results(self):
        for values, sequence in zip(self.input_values, SEQUENCE):
            yield [self._creator.tensor(values, dtype=self._dtype, device=self._device),
                   self._creator.tensor([sequence], dtype=self._dtype, device=self._device)]

    def _create_node(self):
        return RandomNumberNode(lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND, seed=SEED)

    def _extract_results(self, node):
        results = []
        result_one_hot = node.outputs.one_hot_output.tensor.clone()
        result_scalar = node.outputs.scalar_output.tensor.clone()
        results.append(result_one_hot)
        results.append(result_scalar)

        return results

    def _change_node_before_load(self, node: RandomNumberNode):
        node._unit._step = 99
        node._unit._next_generation = 123


def generate_and_validate_sequence(node: RandomNumberNode) -> bool:
    results = []
    for i in range(STEPS):
        node.step()
        output_id = RandomNumberNodeAccessor.get_output_id(node)
        results.append(output_id)

    for i in range(STEPS):
        if results[i] != SEQUENCE[i]:
            return False

    return True


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_node_accessor_and_determinism(device):
    node = RandomNumberNode(lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND, seed=SEED)
    node.allocate_memory_blocks(AllocatingCreator(device))

    # expected sequence
    assert generate_and_validate_sequence(node)

    # expected after re-allocating MBs
    node.allocate_memory_blocks(AllocatingCreator(device))
    assert generate_and_validate_sequence(node)

    # sequence independent on the global seeds
    node.allocate_memory_blocks(AllocatingCreator(device))
    set_global_seeds(None)
    assert generate_and_validate_sequence(node)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_node_accessor_and_determinism_seed(device):

    # new node with a different seed, should produce different results
    node = RandomNumberNode(lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND, seed=None)
    node.allocate_memory_blocks(AllocatingCreator(device))
    assert not generate_and_validate_sequence(node)
