import pytest
import torch
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.random_noise_node import RandomNoiseUnit, RandomNoiseNode, RandomNoiseParams
from tests.core.nodes.node_unit_test_base import NodeTestBase


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_random_noise_unit(device: str):
    shape = [5, 5]
    TestRandomNoise.fix_random_seed()
    creator = AllocatingCreator(device)
    unit = RandomNoiseUnit(creator, shape)
    assert unit.output[0, 0] == 0
    unit.step([])
    assert 0 < unit.output[0, 0] < 1
    big_number = 1000
    added_tensor = creator.ones(shape) * big_number
    unit.step([added_tensor])
    assert big_number < unit.output[0, 0] < big_number + 1


class TestRandomNoise(NodeTestBase):
    """Tests node.

    A minimal test exercising the node.
    """

    _shape: torch.Size = torch.Size((1, 2))

    def _generate_input_tensors(self):
        yield [self._creator.zeros(self._shape, device=self._device, dtype=self._dtype)]

    def _generate_expected_results(self):
        yield []

    def _create_node(self):
        self.fix_random_seed()
        return RandomNoiseNode(params=RandomNoiseParams(list(self._shape)))

    def _check_results(self, expected, results, step):
        for result in results:
            assert 0 < result[0, 0] < 1

    @staticmethod
    def fix_random_seed():
        torch.cuda.manual_seed_all(0)
        torch.manual_seed(0)
