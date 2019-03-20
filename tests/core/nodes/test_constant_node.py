from torchsim.core.nodes.constant_node import ConstantNode
from tests.core.nodes.node_unit_test_base import NodeTestBase


class TestConstantNode:

    class TestConstantNode1(NodeTestBase):
        @classmethod
        def setup_class(cls, device: str = 'cuda'):
            super().setup_class()
            cls._dim = [1, 2, 3]
            cls._const = 10

        def _generate_input_tensors(self):
            yield []

        def _generate_expected_results(self):
            yield [self._creator.full(self._dim, fill_value=self._const, device=self._device, dtype=self._dtype)]

        def _create_node(self):
            return ConstantNode(shape=self._dim, constant=self._const)

    class TestConstantNode2(NodeTestBase):
        @classmethod
        def setup_class(cls, device: str = 'cuda'):
            super().setup_class()
            cls._dim = [5, 5]
            cls._const = 7

        def _generate_input_tensors(self):
            yield []

        def _generate_expected_results(self):
            yield [self._creator.full(self._dim, fill_value=self._const, device=self._device, dtype=self._dtype)]

        def _create_node(self):
            return ConstantNode(shape=self._dim, constant=self._const)

    def test_int_shape(self):
        node1 = ConstantNode((1,))
        node2 = ConstantNode(1)

        assert node1._shape == node2._shape
