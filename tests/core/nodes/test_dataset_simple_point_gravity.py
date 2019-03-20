import pytest
import numpy as np
from dataclasses import astuple

from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.dataset_simple_point_gravity_node import DatasetSimplePointGravityUnit, \
    DatasetSimplePointGravityParams, MoveStrategy
from torchsim.utils.param_utils import Point2D, Size2D


class TestDatasetSimplePointGravityUnit:
    def test_point2D_dataclass(self):
        p = Point2D(10, 5)
        assert (10, 5) == astuple(p)

    @pytest.mark.parametrize('move_strategy, vector, point, expected', [
        (MoveStrategy.DIRECT_TO_POINT, (-1, -1), (1, 1), (0, 0)),
        (MoveStrategy.DIRECT_TO_POINT, (-1, 0), (1, 1), (0, 1)),
        (MoveStrategy.DIRECT_TO_POINT, (-1, 1), (1, 1), (0, 2)),
        (MoveStrategy.DIRECT_TO_POINT, (1, 0), (0, 2), (1, 2)),
        (MoveStrategy.LIMITED_TO_LEFT_DOWN_QUADRANT, (-1, -1), (1, 1), (1, 0)),
        (MoveStrategy.LIMITED_TO_LEFT_DOWN_QUADRANT, (-1, 1), (1, 1), (1, 2)),
        (MoveStrategy.LIMITED_TO_LEFT_DOWN_QUADRANT, (1, -1), (1, 1), (2, 0)),
        (MoveStrategy.LIMITED_TO_LEFT_DOWN_QUADRANT, (1, 1), (1, 1), (2, 1)),
    ])
    def test_move_point(self, move_strategy, vector, point, expected):
        params = DatasetSimplePointGravityParams(Size2D(3, 3), Point2D(1, 1), 1, move_strategy)
        unit = DatasetSimplePointGravityUnit(AllocatingCreator(device='cpu'), params, random=None)
        result = unit._move_point(np.array(vector), Point2D(*point))
        assert Point2D(*expected) == result
