from torchsim.core.nodes.internals.grid_world import GridWorldParams
from torchsim.core.nodes.grid_world_node import GridWorldNode
from tests.core.nodes.node_unit_test_base import NodeTestBase


class TestGridWorld(NodeTestBase):
    params = GridWorldParams('MapA')

    input_values = [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]

    pos = params.agent_pos
    expected_results = [[pos[1] + 0, pos[0] - 1],
                        [pos[1] + 0, pos[0] + 0],
                        [pos[1] + 1, pos[0] + 0],
                        [pos[1] + 0, pos[0] + 0]]

    @classmethod
    def setup_class(cls, device: str = 'cuda'):
        super().setup_class(device)
        cls._dim = 1

    def _generate_input_tensors(self):
        for values in self.input_values:
            yield [self._creator.tensor(values, dtype=self._dtype, device=self._device)]

    def _generate_expected_results(self):
        for values in self.expected_results:
            yield [self._creator.tensor(values, dtype=self._dtype, device=self._device)]

    def _create_node(self):
        return GridWorldNode(self.params)

    def _extract_results(self, node):
        return [node._unit.pos.clone()]
