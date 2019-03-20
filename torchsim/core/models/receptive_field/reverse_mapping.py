import torch
from torchsim.core import get_float, FLOAT_NAN
from torchsim.core.models.receptive_field.grid import Grids
from torch import dtype


class ReverseMapping:
    """
    Receptive field reverse mapping - maps data from parent to child flock
    """
    _parent_map: torch.Tensor
    _float_dtype: dtype
    _grids: Grids
    _device: str
    _data_size: int

    def __init__(self, grids: Grids, device: str, data_size: int):
        self._data_size = data_size
        self._device = device
        self._grids = grids
        self._float_dtype = get_float(self._device)

        # compute indices
        self._parent_map = self._grids.gen_positioned_parent_child_map()

    def reverse_map_concat(self, data: torch.Tensor) -> torch.Tensor:
        """
        Maps data from parent to child expert. Parent positions are distinguished.
        Note: data from multiple parents (on overlaps) are not aggregated, but concatenated instead.

        Args:
            data: tensor of dimensionality (y, x, ...) where:
                * y, x is 2D grid of parents
                * ... is arbitrary number of data dimensions (e.g. context is (2, 2 * context_size))

        Returns:
            Tensor of dimensionality (child_y, child_x, data.shape[0], parent_y, parent_x, *data.shape[1:]) where:
                * child_* is 2D grid of children
                * parent_* is 2D grid of parents positions - Note, this is not grid of parents passed in input!
                    dimensionality of parent positions is equal to the dimensionality of receptive field
                * *data is data received
        """
        data_size = data.shape[2:]  # ignore flock grid (x,y)

        ys = self._parent_map[:, :, :, :, 0].long()
        xs = self._parent_map[:, :, :, :, 1].long()

        # Linearize indices
        i = ys * data.shape[1] + xs

        # Add NaN as last data context and change negative indices to -1 to select NaNs (negative indeces means nans)
        i[i < 0] = -1
        linearized_data = data.view(-1, *data_size)
        linearized_data_with_nans = torch.cat(
            (linearized_data, torch.full((1, *data_size), FLOAT_NAN, device=self._device)), dim=0)
        result = linearized_data_with_nans[i]

        # TODO replace advanced indexing with gather

        return result
