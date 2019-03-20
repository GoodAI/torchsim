from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.mse_node import MseUnit, MseNode
from torchsim.core.utils.tensor_utils import same

from tests.core.nodes.node_unit_test_base import NodeTestBase


class TestMseNode(NodeTestBase):
    @classmethod
    def setup_class(cls, device: str = 'cuda'):
        super().setup_class(device)

    def _generate_input_tensors(self):
        yield [
            self._creator.tensor([0, 1, 2, 0, 1, 2], device=self._device, dtype=self._dtype),
            self._creator.tensor([0, 3, 0, 0, 1, 0], device=self._device, dtype=self._dtype),
        ]

    def _generate_expected_results(self):
        # sum of squared errors will be 12, MSE (divide by 6) will be 2
        yield [self._creator.tensor([2], dtype=self._dtype, device=self._device)]

    def _create_node(self):
        return MseNode(buffer_size=3)


def test_mse_multiple_steps():
    device = 'cpu'
    creator = AllocatingCreator(device=device)
    dtype = creator.float32

    input_tensors1 = [creator.tensor([0, 1, 2, 0, 1, 2], device=device, dtype=dtype),
                      creator.tensor([0, 3, 0, 0, 1, 0], device=device, dtype=dtype)]  # squared diff = 12
    input_tensors2 = [creator.tensor([1, 1, 0, 0, 1, 2], device=device, dtype=dtype),
                      creator.tensor([0, 3, 2, 1, 3, 0], device=device, dtype=dtype)]  # squared diff = 18
    input_tensors3 = [creator.tensor([ 1, 1, 2, 0, 1, 1], device=device, dtype=dtype),
                      creator.tensor([-2, 3, 1, 3, 1, 0], device=device, dtype=dtype)]  # squared diff = 24

    expected_result = creator.tensor([3], dtype=dtype, device=device)  # MSE=3 for a concatenation of the input tensors

    mse_unit = MseUnit(creator, input_shape=input_tensors1[0].shape, buffer_size=3)

    mse_unit.step(input_tensors1[0], input_tensors1[1])
    mse_unit.step(input_tensors2[0], input_tensors2[1])
    mse_unit.step(input_tensors3[0], input_tensors3[1])

    assert same(expected_result, mse_unit._mean_square_error_output)


