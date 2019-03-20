from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple, List, Optional

import torch
from torchsim.core.graph.slot_container import Inputs, MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.gui.observables import ObserverPropertiesItem, disable_on_runtime
from torchsim.gui.validators import validate_positive_with_zero_int, validate_predicate


class SalientRegionUnit(Unit):
    _fixed_region_size: int
    _creator: TensorCreator
    output_coordinates: torch.Tensor

    def __init__(self, creator: TensorCreator, fixed_region_size: int = None):
        super().__init__(creator.device)
        self._fixed_region_size = fixed_region_size
        self._creator = creator
        self.output_coordinates = creator.zeros(4)

    def step(self, saliency_map: torch.Tensor):
        """Computes and outputs the most salient region from the saliency map."""
        self._compute_region_score.cache_clear()
        salient_region = self._find_best_region(saliency_map)
        self._copy_output(salient_region)

    def _copy_output(self, salient_region: Tuple[int, int, int, int]):
        """Copies the salient region coordinates to the output memory block."""
        for i in range(len(salient_region)):
            self.output_coordinates[i] = salient_region[i]

    def _find_best_region(self, saliency_map: torch.Tensor) -> Tuple[int, int, int, int]:
        """Finds the most salient region using a greedy search.

        Roughly:
        1. Partition the saliency map into a grid and pick the highest-scoring cell as the initial region
        2. Iteratively modify the region to improve its score

        Args:
            saliency_map: The gray scale saliency map

        Returns:
            The most salient region as y, x, height, width
        """
        global_mean = saliency_map.mean().item()
        grid_region_scores = self._compute_grid_region_scores(saliency_map)
        initial_region = self._grid_region_from_index(saliency_map, grid_region_scores.argmax().item())
        final_region = self._iterate_best_region(saliency_map, initial_region, global_mean)
        return final_region

    def _compute_grid_region_scores(self, saliency_map: torch.Tensor) -> torch.Tensor:
        """Divides the saliency map into a grid and returns the scores for the grid cells.

        Args:
            saliency_map: The gray scale saliency map

        Returns:
            1D tensor listing the region scores
        """
        return self._creator.tensor(
            [self._compute_region_score(saliency_map, self._grid_region_from_index(saliency_map, i))
             for i in range(self.n_grid_squares(saliency_map))])

    def _grid_region_from_index(self, saliency_map: torch.Tensor, index: int) -> Tuple[int, int, int, int]:
        """Returns the coordinates for a grid region, given its index.

        Args:
            saliency_map: The gray scale saliency map
            index: An index specifying a cell in the grid

        Returns:
            The indexed region as y, x, height, width
        """
        height, width = self.grid_region_height(saliency_map), self.grid_region_width(saliency_map)
        grid_y, grid_x = index // self._n_grid_squares_y(saliency_map), index % self._n_grid_squares_x(saliency_map)
        return grid_y * height, grid_x * width, height, width

    def _iterate_best_region(self,
                             saliency_map: torch.Tensor,
                             initial_region: Tuple[int, int, int, int],
                             global_mean: float = 0) \
            -> Tuple[int, int, int, int]:
        """Tries to find the highest-scoring region through iterative improvement.

        The score of a region in the saliency map is:
            the sum of the values of the covered pixels - (the number of covered pixels * global_mean)

        When comparing regions of different size, the mean value of the saliency map (global_mean) should be provided
        to penalize large regions. When the salient region size is fixed, global_mean can be omitted (set to zero).

        Args:
            saliency_map: The gray scale saliency map
            initial_region: The original region
            global_mean: The mean value in the saliency map; omit when region size is fixed

        Returns:
            The most salient region as y, x, height, width
        """
        current_region = initial_region
        done = False
        while not done:
            best_change, best_change_index \
                = (i.item() for i in self._possible_score_changes(saliency_map, current_region, global_mean).max(dim=0))
            if best_change > 0:
                delta_y, delta_x, delta_height, delta_width = self._region_adjustment_from_index(best_change_index)
                y, x, height, width = current_region
                current_region = y + delta_y, x + delta_x, height + delta_height, width + delta_width
            else:
                done = True
        return current_region

    @lru_cache(maxsize=None)
    def _compute_region_score(self,
                              saliency_map: torch.Tensor,
                              region: Tuple[int, int, int, int],
                              global_mean: float = 0) -> float:
        """Scores the indicated region.

        Args:
            saliency_map: The gray scale saliency map
            region: The region, as y, x, height, width
            global_mean: The mean value in the saliency map; omit if comparing regions of the same size

        Returns:
            The region score
        """
        raw_score = self._select_region(saliency_map, region).sum().item()
        return raw_score - global_mean * self._region_size(region)

    @staticmethod
    def _select_region(saliency_map: torch.Tensor, region: Tuple[int, int, int, int]) -> torch.Tensor:
        """Extracts a rectangular region from the saliency map.

        Args:
            saliency_map: The gray scale saliency map
            region: The region, as y, x, height, width

        Returns:
            A 2D tensor with the selected part of the saliency map
        """
        y, x, height, width = region
        return saliency_map[y:y + height, x:x + width]

    @staticmethod
    def _region_size(region: Tuple[int, int, int, int]):
        """Returns the number of pixels contained in a region.

        Args:
            region: The region as y, x, height, width

        Returns:
            The number of pixels in the region
        """
        y, x, height, width = region
        return height * width

    def _possible_score_changes(self,
                                saliency_map: torch.Tensor,
                                region: Tuple[int, int, int, int],
                                global_mean: float = 0) -> torch.Tensor:
        """Computes the changes in region score resulting from possible changes to the region.

        Args:
            saliency_map: The gray scale saliency map
            region: The original region
            global_mean: The mean value of the saliency map

        Returns:
            1D tensor listing the region score changes
        """
        score_changes = [self._score_change_from_region_adjustment(saliency_map, region,
                                                                   self._region_adjustment_from_index(i),
                                                                   global_mean)
                         for i in range(self.n_possible_region_adjustments)]
        return self._creator.tensor(score_changes)

    def _score_change_from_region_adjustment(self,
                                             saliency_map: torch.Tensor,
                                             region: Tuple[int, int, int, int],
                                             adjustment: Tuple[int, int, int, int],
                                             global_mean: float = 0) -> float:
        """Computes the change in score resulting from a change to a region.

        Args:
            saliency_map: The gray scale saliency map
            region: The original region
            adjustment: The change to the region
            global_mean: The mean value of the saliency map

        Returns:
            The change in score resulting from the adjustment
        """
        y, x, height, width = region
        delta_y, delta_x, delta_height, delta_width = adjustment
        adjusted_region = y + delta_y, x + delta_x, height + delta_height, width + delta_width
        # Check that the suggested change is 'legal' -- inside the map and non-empty
        # If not, return 0 -- in this case, the change will not be selected
        if not self._is_inside_map(saliency_map, adjusted_region) or self._is_empty(adjusted_region):
            return 0
        new_score = self._compute_region_score(saliency_map, adjusted_region, global_mean)
        old_score = self._compute_region_score(saliency_map, region, global_mean)
        return new_score - old_score

    @staticmethod
    def _is_inside_map(saliency_map: torch.Tensor, region: Tuple[int, int, int, int]):
        """Returns true if the region is contained inside the saliency map.

        Args:
            saliency_map: The gray scale saliency map
            region: The region

        Returns:
            True if contained in map
        """
        map_height, map_width = saliency_map.shape
        y, x, height, width = region
        return y >= 0 and x >= 0 and y + height <= map_height and x + width <= map_width

    @staticmethod
    def _is_empty(region: Tuple[int, int, int, int]):
        """Returns true if the region is empty.

        Args:
            region: The region

        Returns:
            True if empty
        """
        y, x, height, width = region
        return height <= 0 or width <= 0

    def _region_adjustment_from_index(self, index: int) -> Tuple[int, int, int, int]:
        """Returns one of the permitted changes to the region coordinates.

        Args:
            index: An index indicating a change to the region

        Returns:
            The change to the region coordinates: delta_y, delta_x, delta_height, delta_width
        """
        if self.has_fixed_region_size:
            return self._region_adjustment_from_index_fixed_size(index)
        else:
            return self._region_adjustment_from_index_variable_size(index)

    @staticmethod
    def _region_adjustment_from_index_fixed_size(index: int) -> Tuple[int, int, int, int]:
        """Returns one of the four permitted changes to the region coordinates, given a fixed region size.

        Args:
            index: An index indicating a change to the region: move up, down, left, or right

        Returns:
            The change to the region coordinates; one of (-1, 0, 0, 0), (1, 0, 0, 0), (0, -1, 0, 0), (0, 1, 0, 0)
        """
        minus_or_plus_one = -1 if index % 2 == 0 else 1
        return (minus_or_plus_one, 0, 0, 0) if index < 2 else (0, minus_or_plus_one, 0, 0)

    @staticmethod
    def _region_adjustment_from_index_variable_size(index: int) -> Tuple[int, int, int, int]:
        """Returns one of the eight permitted changes to the region coordinates, for a variable region size.

        Possible changes to the region are expansion and contraction in each of the corner directions:
        Expand top-left, top-right, bottom-right, bottom-left or contract top-left, top-right, bottom-right, bottom-left

        Args:
            index: An index indicating a change to the region

        Returns:
            The change to the region coordinates; one of (-1, -1, 1, 1), (-1, 0, 1, 1), (0, 0, 1, 1), (0, -1, 1, 1),
            (1, 1, -1, -1), (1, 0, -1, -1), (0, 0, -1, -1), (0, 1, -1, -1)
        """
        delta_y, delta_x = [(-1, -1), (-1, 0), (0, 0), (0, -1), (1, 1), (1, 0), (0, 0), (0, 1)][index]
        delta_height, delta_width = (1, 1) if index < 4 else (-1, -1)  # expand or shrink the region
        return delta_y, delta_x, delta_height, delta_width

    @property
    def has_fixed_region_size(self) -> bool:
        """Returns true if the size of the salient region is fixed."""
        return self._fixed_region_size is not None

    def grid_region_height(self, saliency_map: torch.Tensor) -> int:
        """Returns the height of a cell in the grid used to select the initial region."""
        map_height, map_width = saliency_map.shape
        if self.has_fixed_region_size:
            return self._fixed_region_size
        else:
            return map_height // self._n_grid_squares_y(saliency_map)

    def grid_region_width(self, saliency_map: torch.Tensor) -> int:
        """Returns the width of a cell in the grid used to select the initial region."""
        map_height, map_width = saliency_map.shape
        if self.has_fixed_region_size:
            return self._fixed_region_size
        else:
            return map_width // self._n_grid_squares_x(saliency_map)

    def n_grid_squares(self, saliency_map: torch.Tensor) -> int:
        """Returns the number of cells in the grid used to select the initial region."""
        return self._n_grid_squares_y(saliency_map) * self._n_grid_squares_x(saliency_map)

    def _n_grid_squares_y(self, saliency_map: torch.Tensor) -> int:
        """Height of the 2D grid used to select the initial region."""
        map_height, map_width = saliency_map.shape
        if self.has_fixed_region_size:
            return map_height // self.grid_region_height(saliency_map)
        else:
            return 4

    def _n_grid_squares_x(self, saliency_map: torch.Tensor) -> int:
        """Width of the 2D grid used to select the initial region."""
        map_height, map_width = saliency_map.shape
        if self.has_fixed_region_size:
            return map_width // self.grid_region_width(saliency_map)
        else:
            return 4

    @property
    def n_possible_region_adjustments(self) -> int:
        """Returns the number of possible changes to the region in each step of the greedy search."""
        return 4 if self.has_fixed_region_size else 8


class SalientRegionInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create("Saliency map")


class SalientRegionOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.coordinates = self.create("Region coordinates")

    def prepare_slots(self, unit: SalientRegionUnit):
        self.coordinates.tensor = unit.output_coordinates


@dataclass
class SalientRegionParams(ParamsBase):
    use_fixed_fixed_region_size: bool
    fixed_region_size: int

    def __init__(self, use_fixed_fixed_region_size: bool = False, fixed_region_size: int = 0):
        """Initializes the parameters for the salient region node.

        Args:
            use_fixed_fixed_region_size:
                True if the size of the salient region is fixed by the user; false if it is determined automatically
            fixed_region_size:
                The size the salient region if fixed by the user; a single value since the region is square
        """
        self.use_fixed_fixed_region_size = use_fixed_fixed_region_size
        self.fixed_region_size = fixed_region_size


class SalientRegionNode(WorkerNodeBase[SalientRegionInputs, SalientRegionOutputs]):
    """The salient region node takes a saliency map and selects a salient region.

    The node supports hard attention, where glimpses are selected from a larger input image. It takes a saliency map
    as input and outputs the y, x, height, width coordinates of the most salient region.

    The size of the salient region can be fixed or variable. The saliency map and the salient region are assumed to be
    square (aspect ratio 1:1). This simplifying assumption could be changed if needed.

    See this document for details: https://docs.google.com/document/d/1ljrQyPvo2940yQxVHlxb6Of2e9xhA4rvw39tghGwU2Y
    """
    _params: SalientRegionParams

    def __init__(self, params: Optional[SalientRegionParams] = None, name="SalientRegion"):
        super().__init__(name=name,
                         inputs=SalientRegionInputs(self),
                         outputs=SalientRegionOutputs(self))
        self._params = params.clone() if params else SalientRegionParams()

    def _create_unit(self, creator: TensorCreator) -> Unit:
        return SalientRegionUnit(creator,
                                 fixed_region_size=self.fixed_region_size if self.use_fixed_region_size else None)

    def _step(self):
        self._unit.step(self.inputs.input.tensor)

    def validate(self):
        """Checks that the saliency map has correct dimensions."""
        super().validate()
        saliency_map = self.inputs.input.tensor
        validate_predicate(lambda: saliency_map.dim() == 2,
                           f"The input should be 2D (y, x) but has shape {saliency_map.shape}")
        map_height, map_width = saliency_map.shape
        validate_predicate(lambda: map_height == map_width, "The input saliency map needs to be square")

    @property
    def use_fixed_region_size(self) -> bool:
        return self._params.use_fixed_fixed_region_size

    @use_fixed_region_size.setter
    def use_fixed_region_size(self, value: bool):
        self._params.use_fixed_fixed_region_size = value

    @property
    def fixed_region_size(self) -> int:
        return self._params.fixed_region_size

    @fixed_region_size.setter
    def fixed_region_size(self, value: int):
        validate_positive_with_zero_int(value)
        self._params.fixed_region_size = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
            self._prop_builder.auto('Use fixed region size', type(self).use_fixed_region_size,
                                    edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Fixed region size', type(self).fixed_region_size, edit_strategy=disable_on_runtime)
        ]
