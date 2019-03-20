import torch

from typing import Generator, List, Any
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.nodes.visited_area_node import VisitedAreaNode
from tests.core.nodes.node_unit_test_base import NodeTestBase


class TestVisitedAreaNode(NodeTestBase):
    def setup_class(cls, device: str = 'cuda'):
        super().setup_class()

    node_input = [[[1, 0],
                   [0, 0]],

                  [[0, 1],
                   [0, 0]],

                  [[0, 0],
                   [1, 0]],

                  [[0, 0],
                   [0, 1]]]

    node_output = [[[1., .0],
                    [.0, .0]],

                   [[.9, 1.],
                    [.0, .0]],

                   [[.81, .9],
                    [1., .0]],

                   [[.729, .81],
                    [.9, 1.]]]

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        for step in range(len(self.node_input)):
            yield [self._creator.tensor(self.node_input[step], dtype=self._dtype, device=self._device).unsqueeze(-1)]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        for step in range(len(self.node_output)):
            yield [self._creator.tensor(self.node_output[step], dtype=self._dtype, device=self._device).unsqueeze(-1)]

    def _create_node(self) -> WorkerNodeBase:
        return VisitedAreaNode()

    def _check_results(self, expected, results, step):
        for expected_tensor, result_tensor in zip(expected, results):
            assert self._same(expected_tensor, result_tensor, eps=0.0001)
