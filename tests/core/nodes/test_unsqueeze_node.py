import torch
from typing import Generator, List, Any

from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.nodes import UnsqueezeNode
from tests.core.nodes.node_unit_test_base import NodeTestBase


class TestUnsqueezeNode0(NodeTestBase):
    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        yield [
            self._creator.zeros((2, 1, 3), device=self._device, dtype=self._dtype)
        ]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        yield [
            self._creator.zeros((1, 2, 1, 3), device=self._device, dtype=self._dtype)
        ]

    def _create_node(self) -> WorkerNodeBase:
        return UnsqueezeNode(dim=0)


class TestUnsqueezeNode1(NodeTestBase):
    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        yield [
            self._creator.zeros((2, 1, 3), device=self._device, dtype=self._dtype),
            self._creator.zeros(2, device=self._device, dtype=self._dtype)
        ]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        yield [
            self._creator.zeros((2, 1, 1, 3), device=self._device, dtype=self._dtype),
            self._creator.zeros((2, 1), device=self._device, dtype=self._dtype)
        ]

    def _create_node(self) -> WorkerNodeBase:
        return UnsqueezeNode(dim=1)
