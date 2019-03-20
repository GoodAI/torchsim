import torch

from typing import NamedTuple, Optional

from torchsim.gui.validators import validate_predicate
from torchsim.utils.param_utils import Size2D


class Stride(NamedTuple):
    y: int
    x: int


class Grids:
    """Child and parent level receptive fields.

    Defines the grids of receptive fields at the child and parent level. Supports iteration over receptive
    fields, defining their ordering.

    Example:
        Consider a 6x6 grid of child-level receptive fields. They are indexed like this:

         0  1  2  3  4  5
         6  7  8  9 10 11
        12 13 14 15 16 17
        ...

        Assume that each parent receptive field covers 2x2 child receptive fields. The first parent RF
        corresponds to child RFs 0, 1, 6, 7. The ordering of inputs to the parent level is
        0, 1, 6, 7, 2, 3, 8, 9, 4, ...

    The (y, x) size of the child-level grid is defined in :_child_grid_dims:.
    The number of child receptive fields (or input pixels) covered by a parent-level receptive field is in
    :_parent_receptive_field_dims:.
    """
    _device: str
    _child_grid_dims: Size2D
    _parent_receptive_field_dims: Size2D
    _parent_receptive_field_stride_dims: Stride

    def __init__(self,
                 child_grid_dims: Size2D,
                 parent_rf_dims: Size2D,
                 parent_rf_stride: Optional[Stride] = None,
                 flatten_output_grid_dimensions=False,
                 device: str = 'cpu'):
        self._device = device
        self._check_parameters(child_grid_dims, parent_rf_dims, parent_rf_stride)
        self._child_grid_dims = Size2D(*child_grid_dims)
        self._parent_receptive_field_dims = Size2D(*parent_rf_dims)
        if parent_rf_stride is None:
            self._parent_receptive_field_stride_dims = Stride(*parent_rf_dims)
        else:
            self._parent_receptive_field_stride_dims = Stride(*parent_rf_stride)
        if flatten_output_grid_dimensions:
            self._parent_grid_dims = (self.n_parent_rfs,)
        else:
            self._parent_grid_dims = (self.n_parent_rfs_y, self.n_parent_rfs_x)

    @staticmethod
    def _check_parameters(child_grid_dims: Size2D,
                          parent_rf_dims: Size2D,
                          parent_rf_stride: Stride = None):
        if parent_rf_stride is None:
            parent_rf_stride = parent_rf_dims
        for d in [0, 1]:
            Grids._check_parameters_1d(child_grid_dims[d], parent_rf_dims[d], parent_rf_stride[d])

    @staticmethod
    def _check_parameters_1d(child_grid_dim: int, parent_rf_dim: int, parent_rf_stride: int):
        validate_predicate(lambda: child_grid_dim > 0 and parent_rf_dim > 0 and parent_rf_stride > 0)
        validate_predicate(lambda: parent_rf_stride <= parent_rf_dim)
        validate_predicate(lambda: (child_grid_dim - parent_rf_dim) % parent_rf_stride == 0,
                           additional_message="RFs need to fill the input area exactly")

    def gen_parent_receptive_fields(self):
        for rf_y in range(self.n_parent_rfs_y):
            for rf_x in range(self.n_parent_rfs_x):
                yield (rf_y, rf_x)

    @property
    def child_grid_width(self):
        return self._child_grid_dims.width

    @property
    def child_grid_height(self):
        return self._child_grid_dims.height

    def child_grid_dim(self, dim: int):
        return self._child_grid_dims[dim]

    @property
    def child_grid_area(self):
        height, width = self._child_grid_dims
        return width * height

    @property
    def parent_rf_dims(self):
        return self._parent_receptive_field_dims

    @property
    def parent_rf_width(self):
        return self._parent_receptive_field_dims.width

    @property
    def parent_rf_height(self):
        return self._parent_receptive_field_dims.height

    @property
    def parent_rf_area(self):
        height, width = self._parent_receptive_field_dims
        return width * height

    @property
    def parent_rf_stride_dims(self):
        return self._parent_receptive_field_stride_dims

    def do_parent_rfs_overlap_dim(self, dim: int):
        return self.parent_rf_stride_dims[dim] < self.parent_rf_dims[dim]

    @property
    def n_parent_rfs_y(self):
        return self.n_parent_rfs_dim(0)

    @property
    def n_parent_rfs_x(self):
        return self.n_parent_rfs_dim(1)

    def n_parent_rfs_dim(self, dim: int):
        child_grid_dim = self._child_grid_dims[dim]
        parent_rf_dim = self._parent_receptive_field_dims[dim]
        parent_rf_stride_dim = self._parent_receptive_field_stride_dims[dim]
        return (child_grid_dim - parent_rf_dim) // parent_rf_stride_dim + 1

    @property
    def n_parent_rfs(self):
        return self.n_parent_rfs_y * self.n_parent_rfs_x

    @property
    def number_of_parent_inputs(self):
        return self.n_parent_rfs * self.parent_rf_height * self.parent_rf_width

    @staticmethod
    def _index_from_coordinates(y, x, x_size):
        return y * x_size + x

    @property
    def parent_grid_dims(self):
        return self._parent_grid_dims

    def gen_positioned_parent_child_map(self) -> torch.Tensor:
        """
        Generate map of positioned parent indexes for each child.
        Returns:
            Tensor of dimensions: (child_y, child_x, parent_rf_y, parent_rf_x, 2)
            Where:
                * child_* is dimensionality of child-level
                * parent_rf_* is position of parent (e.g. left/right parents are distinguished)
                * last dimension (2) holds 2D coordinates of the parent
        Examples: see test
        """
        nan = -1
        rc_dims = self._parent_receptive_field_dims
        result = torch.full((*self._child_grid_dims, *rc_dims, 2), nan, dtype=torch.int64,
                            device=self._device)
        for (py, px) in self.gen_parent_receptive_fields():
            rf_x = px * self._parent_receptive_field_stride_dims.x
            rf_y = py * self._parent_receptive_field_stride_dims.y
            for y in range(rc_dims.height):
                for x in range(rc_dims.width):
                    child_x = rf_x + x
                    child_y = rf_y + y
                    rc_x = rc_dims.width - x - 1
                    rc_y = rc_dims.height - y - 1
                    result[child_y, child_x, rc_y, rc_x, 0] = py
                    result[child_y, child_x, rc_y, rc_x, 1] = px

        return result
