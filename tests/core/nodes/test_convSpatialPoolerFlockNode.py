from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import ConvSpatialPoolerFlockNode
from tests.core.nodes.node_unit_test_base import NodeTestBase, AnyResult


class TestConvSpatialPoolerFlockNode(NodeTestBase):
    """Check that the flock learns to recognize 3 clusters"""
    experts_grid = (1, 2)
    image_size = (3,)
    params = ExpertParams()
    params.n_cluster_centers = 3
    params.flock_size = experts_grid[0] * experts_grid[1]
    params.spatial.batch_size = 2
    params.spatial.learning_period = 1
    step_pairs_not_comparing = 4

    @classmethod
    def setup_class(cls, device: str = 'cuda'):
        super().setup_class()

        cls.inputs = cls._creator.tensor([[[[3.7, 14, 56], [-12, -11, 0]]],
                                          [[[-12, -11, 0], [1, 2, 3]]]], device=cls._device, dtype=cls._dtype)
        cls.expected_results = cls._creator.tensor([[[[1, 0, 0], [0, 1, 0]]],
                                                    [[[0, 1, 0], [0, 0, 1]]]], device=cls._device, dtype=cls._dtype)
        cls.n_inputs = 2

    def _generate_input_tensors(self):
        for step in range(self.step_pairs_not_comparing + 1):
            yield [self.inputs[0]]
            yield [self.inputs[1]]

    def _generate_expected_results(self):
        for step in range(self.step_pairs_not_comparing):
            yield [AnyResult, AnyResult]

        yield [self.expected_results[0], AnyResult]
        yield [self.expected_results[1], AnyResult]

    def _create_node(self):

        return ConvSpatialPoolerFlockNode(params=self.params, seed=8)
