import torch

from torchsim.gui.observers.cluster_observer import ClusterObserver
from torchsim.core.nodes.flatten_node import FlattenNode
from tests.core.nodes.node_unit_test_base import NodeTestBase


class TestFlattenNode0(NodeTestBase):
    def _generate_input_tensors(self):
        yield [
            self._creator.zeros((2, 3, 4, 5, 6), device='cuda', dtype=torch.float32)
        ]

    def _generate_expected_results(self):
        yield [
            self._creator.zeros((2, 60, 6), device='cuda', dtype=torch.float32)
        ]

    def _create_node(self):
        return FlattenNode(start_dim=1, end_dim=3)


class TestFlattenNode1(NodeTestBase):
    def _generate_input_tensors(self):
        yield [
            self._creator.zeros((2, 3, 4, 5, 6), device='cuda', dtype=torch.float32)
        ]

    def _generate_expected_results(self):
        yield [
            self._creator.zeros((2, 60, 6), device='cuda', dtype=torch.float32)
        ]

    def _create_node(self):
        return FlattenNode(start_dim=1, end_dim=-2)


class TestFlattenNode2(NodeTestBase):
    def _generate_input_tensors(self):
        yield [
            self._creator.zeros((2, 3, 4, 5, 6), device='cuda', dtype=torch.float32)
        ]

    def _generate_expected_results(self):
        yield [
            self._creator.zeros((2, 3, 4, 5, 6), device='cuda', dtype=torch.float32)
        ]

    def _create_node(self):
        return FlattenNode(start_dim=-5, end_dim=0)


class TestFlattenNode3(NodeTestBase):
    def _generate_input_tensors(self):
        yield [
            self._creator.zeros((2, 3, 4, 5, 6), device='cuda', dtype=torch.float32)
        ]

    def _generate_expected_results(self):
        yield [
            self._creator.zeros((720,), device='cuda', dtype=torch.float32)
        ]

    def _create_node(self):
        return FlattenNode(start_dim=-5, end_dim=4)


class TestFlattenNode4(NodeTestBase):
    def _generate_input_tensors(self):
        yield [
            self._creator.zeros((2, 3, 4, 5, 6), device='cuda', dtype=torch.float32)
        ]

    def _generate_expected_results(self):
        yield [
            self._creator.zeros((24, 5, 6), device='cuda', dtype=torch.float32)
        ]

    def _create_node(self):
        return FlattenNode(start_dim=0, end_dim=2)


class TestFlattenNode5(NodeTestBase):
    def _generate_input_tensors(self):
        yield [
            self._creator.zeros((2, 3, 4, 5, 6), device='cuda', dtype=torch.float32)
        ]

    def _generate_expected_results(self):
        yield [
            self._creator.zeros((720,), device='cuda', dtype=torch.float32)
        ]

    def _create_node(self):
        return FlattenNode()

