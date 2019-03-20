import torch

from torchsim.core import get_float
from torchsim.core.models.receptive_field.grid import Grids


class Mapping:
    """Mapping from child expert output or from an image to parent expert input.

    The mapper converts
        a. the output from a set of child experts to input to a set of parent experts ("expert input"), or
        b. an input image with multiple channels to input to a set of experts ("sensory input").
    The experts correspond to rectangular areas (receptive fields or RFs) within the field of view (FOV).

    The mapping is needed because the input to each parent expert needs to be contiguous and this means
    that the data need to be ordered differently depending on how the receptive fields are laid out.

    In the simple case, each child expert has exactly one parent, the size of child output equal the
    size of parent input, and each chunk of data from the child output appears exactly once in the
    parent input. However, if parent RFs are overlapping, children will have multiple parents, and
    parent input will be larger than the child output as chunks of the child output will be repeated for
    different parents.

    We currently assume the following formats:

        child expert output:    Y X C
        sensory input:          Y X C
        parent expert input:    E Y_local X_local C

    where E = expert and C = child output or channels. YXC ordering is used internally as a canonical
    format.
    """
    _grids: Grids
    _child_output_size: int
    _n_channels: int
    _indexes: torch.int64
    _occupancies: torch.Tensor
    _pre_allocated_parent_input: torch.Tensor
    _pre_allocated_back_projection: torch.Tensor = None
    _device: str

    def __init__(self, grids: Grids, device: str, child_output_size: int = None, n_channels: int = None):
        assert child_output_size is None and n_channels is not None \
               or child_output_size is not None and n_channels is None, \
            "Exactly one of child_output_size or n_channels must be provided"
        self._grids = grids
        self._child_output_size = child_output_size
        self._n_channels = n_channels
        self._device = device
        self._float_dtype = get_float(self._device)
        self._indexes = self._compute_indexes()
        self._occupancies = self._compute_occupancies()
        self._pre_allocated_parent_input = torch.zeros(grids.n_parent_rfs * grids.parent_rf_area * self.chunk_size,
                                                       dtype=self._float_dtype, device=self._device)

    def __call__(self, child_output: torch.Tensor) -> torch.Tensor:
        return self.map(child_output)

    @classmethod
    def from_default_input(cls, grids: Grids, device: str, chunk_size: int = 1):
        """Creates a mapping form the default input format to parent expert input.

        Since we assume YXC format for both sensory input and child experts, this can be used in both cases.
        """
        return cls.from_sensory_input(grids, device, chunk_size)

    @classmethod
    def from_sensory_input(cls, grids: Grids, device: str, n_channels: int = 1):
        """Creates a mapping from sensory input to parent expert input."""
        return cls(grids, child_output_size=None, n_channels=n_channels, device=device)

    @classmethod
    def from_child_expert_output(cls, grids: Grids, device: str, child_output_size: int = 1, ):
        """Creates a mapping from child expert output to parent expert input."""
        return cls(grids, child_output_size=child_output_size, n_channels=None, device=device)

    @property
    def using_sensory_input(self):
        return self._n_channels is not None

    @property
    def chunk_size(self):
        return self._n_channels if self.using_sensory_input else self._child_output_size

    @property
    def parent_dimensions(self):
        return torch.Size([self._grids.n_parent_rfs,
                           self._grids.parent_rf_height,
                           self._grids.parent_rf_width,
                           self.chunk_size])

    def map(self, child_output: torch.Tensor) -> torch.Tensor:
        """Maps the child level output (from sensory input or child expert output) to parent expert input."""
        self._check_child_output_dimensions(child_output)
        reordered = self._project(child_output, self._pre_allocated_parent_input)
        return self._shape_parent_expert_input(reordered)

    def inverse_map(self, parent_input: torch.Tensor) -> torch.Tensor:
        """Performs the back projection or inverse mapping.

        From parent expert input to child level output.
        """
        self._check_parent_input_dimensions(parent_input)

        size = self._grids.child_grid_area * self.chunk_size
        result = torch.zeros(size, dtype=self._float_dtype, device=self._device)

        reordered = self._project(parent_input, result, forward=False)
        return self._shape_back_projection(reordered)

    def _project(self, elements: torch.Tensor, destination: torch.Tensor, forward: bool = True) -> torch.Tensor:
        flat = Mapping._flatten(elements)
        # Use a gather or scatter type operation depending on the direction.
        # This way we can use the same index matrix for both directions.
        if forward:
            return torch.gather(flat, 0, self._indexes, out=destination)
        else:
            destination.zero_()
            destination.put_(self._indexes, flat, accumulate=True)
            return destination.div_(self._occupancies)

    @staticmethod
    def _check_child_output_dimensions(child_output: torch.Tensor):
        assert child_output.dim() <= 3

    @staticmethod
    def _check_parent_input_dimensions(parent_input: torch.Tensor):
        assert parent_input.dim() >= 2

    @staticmethod
    def _flatten(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(-1)

    def _shape_parent_expert_input(self, parent_expert_input: torch.Tensor) -> torch.Tensor:
        dimensions = self._grids.parent_grid_dims + \
                     (self._grids.parent_rf_height, self._grids.parent_rf_width, self.chunk_size)
        return parent_expert_input.view(dimensions)

    def _shape_back_projection(self, back_projection: torch.Tensor) -> torch.Tensor:
        if self.chunk_size == 1:
            return back_projection.view(self._grids.child_grid_height,
                                        self._grids.child_grid_width)
        else:
            return back_projection.view(self._grids.child_grid_height,
                                        self._grids.child_grid_width,
                                        self.chunk_size)

    def _compute_indexes(self) -> torch.int64:
        """Computes the tensor of indexes defining the mapping from the lower layer to parent input.

        The bottom layer consists of child experts or of sensory input.
        """
        tensor_size = self._grids.child_grid_area * self.chunk_size
        initial_indexes = torch.arange(0, tensor_size, dtype=torch.int64, device=self._device)
        initial_indexes = self._permute_index_dimensions(initial_indexes)
        initial_indexes = self._repeat_overlaps(initial_indexes)

        shaped = initial_indexes.reshape(self._grids.n_parent_rfs_y,
                                         self._grids.parent_rf_height,
                                         self._grids.n_parent_rfs_x,
                                         self._grids.parent_rf_width,
                                         self.chunk_size)
        rearranged = shaped.permute(0, 2, 1, 3, 4)

        return self._flatten(rearranged)

    def _compute_occupancies(self) -> torch.Tensor:
        """Computes the occupancy matrix.

        The matrix is the outer product of the occupancy vectors for each dimension.
        """
        occupancies = self._occupancies_1d(0).ger(self._occupancies_1d(1))
        if self.chunk_size > 1:
            occupancies = occupancies.unsqueeze(2).expand(-1, -1, self.chunk_size)
        return self._flatten(occupancies)

    def _occupancies_1d(self, dim: int):
        indexes = self._indexes_1d(dim)
        counts = self._counts(indexes)
        return counts[0:self._grids.child_grid_dim(dim)]

    def _counts(self, values: torch.int64) -> torch.Tensor:
        """Count the number of times each index value occurs."""
        size = values.numel()
        ones = torch.ones(size, dtype=self._float_dtype, device=self._device)
        counts = torch.zeros(size, dtype=self._float_dtype, device=self._device)
        counts.put_(values, ones, accumulate=True)
        return counts

    def _permute_index_dimensions(self, indexes: torch.int64) -> torch.int64:
        """Rearranges indexes to allow different orderings of input dimensions.

        Currently does nothing, since we are using YXC for everything.
        """
        return indexes

    def _repeat_overlaps(self, indexes: torch.int64) -> torch.int64:
        """Expands index tensor to handle overlapping receptive fields."""
        indexes = indexes.reshape(self._grids.child_grid_height, self._grids.child_grid_width, self.chunk_size)
        for dim in range(0, 2):
            indexes = self._repeat_overlaps_1d(indexes, dim)
        return indexes

    def _repeat_overlaps_1d(self, indexes: torch.int64, dim: int) -> torch.int64:
        assert dim <= 1, "We can only handle two dimensions"

        if self._grids.do_parent_rfs_overlap_dim(dim):
            indexes = indexes.index_select(dim, self._indexes_1d(dim))

        return indexes

    def _indexes_1d(self, dim: int):
        rf_size = self._grids.parent_rf_dims[dim]
        stride = self._grids.parent_rf_stride_dims[dim]
        n_rfs = self._grids.n_parent_rfs_dim(dim)

        # return torch.int64([i for i in self._generate_1d_indexes(rf_size, stride, n_rfs)], device=self._device)
        return torch.tensor([i for i in self._generate_1d_indexes(rf_size, stride, n_rfs)],
                            dtype=torch.int64,
                            device=self._device)

    @staticmethod
    def _generate_1d_indexes(rf_size: int, stride: int, n_rfs: int):
        for rf in range(0, n_rfs):
            for local_coordinate in range(0, rf_size):
                yield rf * stride + local_coordinate

    def _from_cyx(self, indexes: torch.int64) -> torch.int64:
        """Converts sensory input from CYX format to YXC."""
        indexes = indexes.view(self._n_channels,
                               self._grids.child_grid_height,
                               self._grids.child_grid_width)
        indexes = indexes.permute(1, 2, 0)
        indexes = self._flatten(indexes)

        return indexes
