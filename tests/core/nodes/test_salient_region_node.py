from typing import Generator, List, Any

import torch
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.nodes.salient_region_node import SalientRegionNode, SalientRegionParams
from tests.core.nodes.node_unit_test_base import NodeTestBase


class TestSalientRegionNode(NodeTestBase):
    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        input_size = 10
        saliency_map = self._creator.zeros(input_size, input_size, dtype=torch.float)
        saliency_map[4, 4] = 1
        yield [saliency_map]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        y, x, height, width = 4, 4, 1, 1
        yield [self._creator.tensor([y, x, height, width], dtype=torch.float)]

    def _create_node(self) -> WorkerNodeBase:
        return SalientRegionNode()


class TestFixedSizeSalientRegionNode(TestSalientRegionNode):
    _fixed_region_size = 2

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        y, x, height, width = 4, 4, self._fixed_region_size, self._fixed_region_size
        yield [self._creator.tensor([y, x, height, width], dtype=torch.float)]

    def _create_node(self) -> WorkerNodeBase:
        params = SalientRegionParams(use_fixed_fixed_region_size=True, fixed_region_size=self._fixed_region_size)
        return SalientRegionNode(params)
