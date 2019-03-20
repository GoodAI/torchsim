import logging
import torch
from abc import abstractmethod, ABC
from enum import Enum, auto

from torchsim.core.exceptions import IllegalArgumentException
from torchsim.core.graph.hierarchical_observable_node import HierarchicalObservableNode
from torchsim.core.models.spatial_pooler import SPFlockLearning
from torchsim.core.utils.inverse_projection_utils import get_inverse_projections_for_all_clusters
from torchsim.gui.observables import ObserverPropertiesItemSelectValueItem, ObserverPropertiesBuilder, \
    ObserverPropertiesItemState
from torchsim.gui.observer_system import ObserverPropertiesItem, ClusterObservable
from typing import NamedTuple, TYPE_CHECKING

from torchsim.gui.observers.flock_process_observable import FlockProcessObservable
from torchsim.gui.observers.memory_block_observer import is_valid_tensor
from torchsim.gui.observers.tensor_observable import TensorViewProjection
from torchsim.gui.ui_utils import encode_image
from torchsim.gui.validators import *

if TYPE_CHECKING:
    from torchsim.core.nodes.expert_node import ExpertFlockNode
    from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode

logger = logging.getLogger(__name__)

FloatDataType = Union[List[List[float]], None]
IntDataType = Union[List[List[int]], None]


class ClusterDatapointsData(NamedTuple):
    cluster_datapoints: FloatDataType
    cluster_datapoints_cluster_ids: IntDataType


class PcaData(NamedTuple):
    cluster_centers: FloatDataType
    cluster_datapoints: Optional[ClusterDatapointsData]


class ClusterCentersData(NamedTuple):
    current_id: int
    current_size_coef: float
    size_coef: float
    projections: Optional[List[str]]


class FDsimData(NamedTuple):
    iterations_per_step: int
    reset: bool


class SpringLinesData(NamedTuple):
    pass


class SplineArrowsData(NamedTuple):
    significances: Optional[List[float]]
    significance_root_power: float
    size_coef: float
    inner_shift_coef: float
    outer_shift_coef: float


class SequencesData(NamedTuple):
    sequences: FloatDataType
    occurrences: Optional[List[float]]


class ClusterObserverData(NamedTuple):
    cluster_centers: ClusterCentersData
    spring_lines: SpringLinesData
    spline_arrows: SplineArrowsData
    fdsim: FDsimData
    # cluster_similarities: FloatDataType
    sequences: SequencesData
    n_dims: int
    n_cluster_centers: int
    n_sequences: int
    sequence_length: int
    pca: PcaData
    projection_type: str
    width: int
    height: int


class ClusterObserverProjection(Enum):
    PCA = 1
    FD_SIM = 2


class ClusterObserverClusterSimilaritiesAlgorithm(Enum):
    NEIGHBOURS = 1
    ANYWHERE = 2


class SequenceSignificanceSource(Enum):
    MODEL_FREQUENCY = auto()
    SEQ_LIKELIHOODS_ACTIVE = auto()
    SEQ_LIKELIHOODS_CLUSTERS = auto()
    SEQ_LIKELIHOODS_EXPLORATION = auto()
    SEQ_LIKELIHOODS_GOAL_DIRECTED = auto()
    SEQ_LIKELIHOODS_PRIORS_CLUSTERS = auto()
    SEQ_LIKELIHOODS_PRIORS_CLUSTERS_CONTEXT = auto()


class DataTransformer:
    @abstractmethod
    def project(self, data: torch.Tensor, n_dims: int) -> torch.Tensor:
        pass


class PcaTransformer(DataTransformer):
    _projection_matrix: torch.Tensor
    _means: torch.Tensor
    _std_devs: torch.Tensor

    _computed: bool

    def __init__(self):
        self._computed = False

    def is_computed(self) -> bool:
        return self._computed

    @staticmethod
    def normalize(data: torch.Tensor, means: torch.Tensor, std_devs: torch.Tensor):
        return data.clone().sub_(means).div_(std_devs)

    @staticmethod
    def compute_explained_variances(singular_values: torch.Tensor, n_examples: int) -> List[int]:
        eigen_values = singular_values * singular_values / (n_examples - 1)
        total = eigen_values.sum()
        explained_variances = eigen_values[:3].cumsum(dim=0) * (100 / total)
        return explained_variances.cpu().tolist()

    def compute(self, data: torch.Tensor) -> List[int]:
        self._means = data.mean(dim=0)
        self._std_devs = data.std(dim=0, unbiased=True)
        self._std_devs[self._std_devs == 0] = 1
        data_normalized = PcaTransformer.normalize(data, self._means, self._std_devs)
        _, S, V = data_normalized.svd(some=True)
        # TODO - handle low dimensionality exception here - extends S and V instead of handling in project method
        self._projection_matrix = V[:, :3]
        self._computed = True
        return PcaTransformer.compute_explained_variances(S, data.shape[0])

    def project(self, data: torch.Tensor, n_dims: int) -> torch.Tensor:
        data_normalized = PcaTransformer.normalize(data, self._means, self._std_devs)
        projected = data_normalized.mm(self._projection_matrix)
        data_size = projected.shape[-1]
        if data_size < 3:
            # handle correctly data with dimensionality < 3
            missing_dims = 3 - data_size
            return torch.cat((projected, projected.new_zeros((projected.shape[0], missing_dims))), 1)
        elif n_dims == 2:
            projected[:, 2] = 0
        return projected


class ClusterObserverDataBuilderBase(ABC):
    _prop_builder: ObserverPropertiesBuilder
    _tensor_provider: 'TensorProvider'

    def __init__(self, tensor_provider: 'TensorProvider'):
        self._prop_builder = ObserverPropertiesBuilder(self)
        self._tensor_provider = tensor_provider


class ClusterCentersDataBuilder(ClusterObserverDataBuilderBase):
    _current_size_coef: float
    _size_coef: float
    _group_id: int
    _show_projections: bool

    def __init__(self, tensor_provider: 'TensorProvider'):
        super().__init__(tensor_provider)
        self._current_size_coef = 2.0
        self._size_coef = 1.0
        self._group_id = 0
        self._show_projections = True

    @property
    def size_coef(self) -> float:
        return self._size_coef

    @size_coef.setter
    def size_coef(self, value: float):
        validate_positive_float(value)
        self._size_coef = value

    @property
    def current_size_coef(self) -> float:
        return self._current_size_coef

    @current_size_coef.setter
    def current_size_coef(self, value: float):
        validate_positive_float(value)
        self._current_size_coef = value

    @property
    def group_id(self) -> int:
        return self._group_id

    @group_id.setter
    def group_id(self, value: int):
        # TODO: validate value is in range
        self._group_id = value

    @property
    def show_projections(self) -> bool:
        return self._show_projections

    @show_projections.setter
    def show_projections(self, value: bool):
        self._show_projections = value

    def get_data(self) -> ClusterCentersData:
        current_id = self._tensor_provider.current_cluster_id()
        return ClusterCentersData(current_id, self.current_size_coef, self.size_coef,
                                  self._tensor_provider.cluster_center_projections(self._group_id)
                                  if self._show_projections else None)

    def get_properties(self, enabled: bool) -> List[ObserverPropertiesItem]:
        return [
                   self._prop_builder.auto("Current cc size coef", type(self).current_size_coef, enabled=enabled),
                   self._prop_builder.auto("Size coef", type(self).size_coef, enabled=enabled),
                   self._prop_builder.auto("Group id", type(self).group_id, enabled=enabled),
                   self._prop_builder.auto("Show Projections", type(self).show_projections, enabled=enabled)

               ] + self._tensor_provider.cluster_center_projections_properties()


class FDsimDataBuilder(ClusterObserverDataBuilderBase):
    """Force Directed Simulation projection."""
    _iterations_per_step: int = 25
    _reset: bool = False

    def get_data(self) -> FDsimData:
        data = FDsimData(self._iterations_per_step, self._reset)
        # Reset is sent just once
        self._reset = False
        return data

    def get_properties(self) -> List[ObserverPropertiesItem]:
        def update_iterations_per_step(value):
            self._iterations_per_step = int(value)
            return value

        return [
            ObserverPropertiesItem('Iterations per step', 'number', self._iterations_per_step,
                                   update_iterations_per_step),
        ]

    def reset(self):
        self._reset = True


class ClusterDatapointsDataBuilder(ClusterObserverDataBuilderBase):

    def get_data(self, transformer: DataTransformer, n_dims: int) -> ClusterDatapointsData:
        return ClusterDatapointsData(
            cluster_datapoints=self._get_cluster_datapoints(transformer, n_dims),
            cluster_datapoints_cluster_ids=self._get_cluster_datapoints_ids()
        )

    def _get_cluster_datapoints(self, transformer: DataTransformer, n_dims: int) -> IntDataType:
        tensor = self._tensor_provider.learn_process_data_batch()
        if tensor is None:
            return None
        tensor = tensor.clone().masked_fill_(torch.isnan(tensor), 0)
        result = transformer.project(tensor.view(tensor.shape[0], -1), n_dims)
        return result.cpu().tolist()

    def _get_cluster_datapoints_ids(self) -> IntDataType:
        tensor = self._tensor_provider.learn_process_last_batch_clusters()
        if tensor is None:
            return None
        return tensor.cpu().view(tensor.shape[0], -1).argmax(dim=1).tolist()


class PcaDataBuilder(ClusterObserverDataBuilderBase):
    _cluster_datapoints: ClusterDatapointsDataBuilder
    _reset: bool
    # _pca_params: PcaParams
    _pca_transformer: PcaTransformer

    def __init__(self, tensor_provider: 'TensorProvider'):
        super().__init__(tensor_provider)
        self._reset = True
        self._pca_transformer = PcaTransformer()
        self._cluster_datapoints = ClusterDatapointsDataBuilder(tensor_provider)

    def _update_pca_transformer(self):
        if not self._reset:
            return None

        tensor = self._tensor_provider.learn_process_data_batch()
        if tensor is None:
            return None

        self._reset = False
        tensor = tensor.clone().masked_fill_(torch.isnan(tensor), 0)
        explained_vars = self._pca_transformer.compute(tensor)
        if len(explained_vars) < 3:
            logger.info(f'PCA computed')
        else:
            logger.info(
                f'PCA computed, explained variance 2D: {explained_vars[1]:.2f} %, 3D: {explained_vars[2]:.2f} %')

    def get_data(self, n_dims: int, show_datapoints: bool) -> PcaData:
        self._update_pca_transformer()
        # self._try_compute_pca(n_dims)
        if self._pca_transformer.is_computed():
            return PcaData(
                # calculation_data=self._get_calculation_data(),
                cluster_centers=self._get_cluster_centers(n_dims),
                cluster_datapoints=self._cluster_datapoints.get_data(self._pca_transformer,
                                                                     n_dims) if show_datapoints else None
            )
        else:
            return PcaData(None, None)

    def _get_cluster_centers(self, n_dims: int) -> FloatDataType:
        if self._reset:
            return None

        expert_data = self._tensor_provider.sp_cluster_centers()
        if expert_data is None:
            return None
        result = self._pca_transformer.project(expert_data.view(expert_data.shape[0], -1), n_dims)
        return result.cpu().tolist()

    def reset(self):
        self._reset = True


class SpringLinesBuilder(ClusterObserverDataBuilderBase):

    def get_data(self) -> SpringLinesData:
        return SpringLinesData()


class SplineArrowsBuilder(ClusterObserverDataBuilderBase):
    _sequence_significance_source: SequenceSignificanceSource
    _significance_root_power: float
    _size_coef: float
    _inner_shift_coef: float
    _outer_shift_coef: float

    def __init__(self, tensor_provider: 'TensorProvider'):
        super().__init__(tensor_provider)
        self._sequence_significance_source = SequenceSignificanceSource.MODEL_FREQUENCY
        self._significance_root_power = 1.0
        self._size_coef = 1.0
        self._inner_shift_coef = 0.25
        self._outer_shift_coef = 0.05

    @property
    def sequence_significance_source(self) -> SequenceSignificanceSource:
        return self._sequence_significance_source

    @sequence_significance_source.setter
    def sequence_significance_source(self, value: SequenceSignificanceSource):
        self._sequence_significance_source = value

    @property
    def size_coef(self) -> float:
        return self._size_coef

    @size_coef.setter
    def size_coef(self, value: float):
        validate_positive_float(value)
        self._size_coef = value

    @property
    def significance_root_power(self) -> float:
        return self._significance_root_power

    @significance_root_power.setter
    def significance_root_power(self, value: float):
        validate_positive_float(value)
        self._significance_root_power = value

    @property
    def inner_shift_coef(self) -> float:
        return self._inner_shift_coef

    @inner_shift_coef.setter
    def inner_shift_coef(self, value: float):
        validate_float_in_range(value, 0., 0.5)
        self._inner_shift_coef = value

    @property
    def outer_shift_coef(self) -> float:
        return self._outer_shift_coef

    @outer_shift_coef.setter
    def outer_shift_coef(self, value: float):
        validate_float_in_range(value, 0., 0.25)
        self._outer_shift_coef = value

    def get_data(self) -> SplineArrowsData:
        significances = self._tensor_provider.tp_sequence_significance(self._sequence_significance_source)
        significances_list = significances.tolist() if significances is not None else None
        return SplineArrowsData(significances_list,
                                self._significance_root_power,
                                self._size_coef,
                                self._inner_shift_coef,
                                self._outer_shift_coef)

    def get_properties(self, enabled: bool) -> List[ObserverPropertiesItem]:
        return [
            self._prop_builder.auto("Significance Source", type(self).sequence_significance_source, enabled=enabled),
            self._prop_builder.auto("Significance Root Power", type(self).significance_root_power, enabled=enabled),
            self._prop_builder.auto("Size", type(self).size_coef, enabled=enabled),
            self._prop_builder.auto("Inner Shift", type(self).inner_shift_coef, enabled=enabled),
            self._prop_builder.auto("Outer Shift", type(self).outer_shift_coef, enabled=enabled),
        ]


class SequencesBuilder(ClusterObserverDataBuilderBase):

    def get_data(self) -> SequencesData:
        if not self._tensor_provider.has_temporal_pooler():
            return SequencesData(None, None)

        sequences = self._tensor_provider.tp_frequent_seqs()
        occurrences = self._tensor_provider.tp_frequent_seq_occurrences()

        sequences = sequences.tolist() if sequences is not None else sequences
        occurrences = occurrences.tolist() if occurrences is not None else occurrences

        return SequencesData(sequences, occurrences)


# class ClusterSimilaritiesBuilder(ClusterObserverDataBuilderBase):
#     _similarities: torch.Tensor = None
#     _last_learning: int = -1
#     _force_compute: bool = False
#
#     def get_data(self, algorithm: ClusterObserverClusterSimilaritiesAlgorithm) -> FloatDataType:
#         return self._get_cluster_similarities(algorithm)
#
#     def _get_cluster_similarities(self, algorithm: ClusterObserverClusterSimilaritiesAlgorithm):
#         if not self._tensor_provider.has_temporal_pooler():
#             return None
#
#         cluster_count = self._tensor_provider.n_cluster_centers()
#
#         # Compute similarities only after learning
#         last_learning = self._tensor_provider.sp_execution_counter_learning().tolist()
#         if self._last_learning < last_learning or self._similarities is None or self._force_compute:
#             self._force_compute = False
#             self._last_learning = last_learning
#             # Compute similarities
#
#             sequences = self._tensor_provider.tp_frequent_seqs()
#             occurrences = self._tensor_provider.tp_frequent_seq_occurrences()
#
#             if sequences is None or occurrences is None:
#                 return None
#
#             if algorithm == ClusterObserverClusterSimilaritiesAlgorithm.NEIGHBOURS:
#                 self._similarities = ClusterUtils.compute_similarities(cluster_count, sequences, occurrences)
#             elif algorithm == ClusterObserverClusterSimilaritiesAlgorithm.ANYWHERE:
#                 self._similarities = ClusterUtils.compute_similarities_orderless(cluster_count, sequences, occurrences)
#             else:
#                 raise IllegalArgumentException(f'Unknown algorithm {algorithm}')
#
#         return self._similarities.tolist()
#
#     def recompute(self):
#         self._force_compute = True


class ClusterProjectionsGroupProperties:
    _prop_builder: ObserverPropertiesBuilder
    _is_rgb: bool = False
    tensor_view_projection: TensorViewProjection

    def __init__(self):
        self._prop_builder = ObserverPropertiesBuilder(self)
        self.tensor_view_projection = TensorViewProjection(is_buffer=False)

    def project_and_scale(self, tensor: torch.Tensor):
        tensor, projection_params = self.tensor_view_projection.transform_tensor(tensor, self.is_rgb)
        return tensor, projection_params

    @property
    def is_rgb(self) -> bool:
        return self._is_rgb

    @is_rgb.setter
    def is_rgb(self, value: bool):
        self._is_rgb = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        properties = [
                         self._prop_builder.auto("RGB", type(self).is_rgb)
                     ] + self.tensor_view_projection.get_properties()

        header_name = f'Projections'
        for prop in properties:
            prop.name = f"{header_name}.{prop.name}"

        return [
            self._prop_builder.collapsible_header(header_name, True),
            *properties]


class TensorProvider(ABC):

    def __init__(self, node: HierarchicalObservableNode, expert_id: int):
        self._cluster_projections_group_properties = ClusterProjectionsGroupProperties()
        self._node = node
        self._expert_id = expert_id

    @abstractmethod
    def n_cluster_centers(self) -> int:
        pass

    @abstractmethod
    def n_sequences(self) -> int:
        pass

    @abstractmethod
    def sequence_length(self) -> int:
        pass

    @abstractmethod
    def has_temporal_pooler(self) -> bool:
        pass

    @abstractmethod
    def current_cluster_id(self) -> int:
        pass

    def learn_process_data_batch(self):
        process, idx = self._learn_process_with_expert_idx()
        return process.data_batch[idx] if process is not None else None

    def learn_process_last_batch_clusters(self):
        process, idx = self._learn_process_with_expert_idx()
        if process is not None and process.last_batch_clusters is not None:
            return process.last_batch_clusters[idx]
        # tensor = self.flock_node.memory_blocks.sp.buffer_clusters.tensor
        # if ClusterDatapointsData.tensor_is_invalid(tensor):
        #     return None
        # tensor = tensor[self.expert_id]
        # return tensor.cpu().view(tensor.size(0), -1).argmax(dim=1).tolist()

    @abstractmethod
    def sp_cluster_centers(self) -> Union[torch.Tensor, None]:
        pass

    @abstractmethod
    def sp_execution_counter_learning(self) -> Union[torch.Tensor, None]:
        pass

    @abstractmethod
    def tp_frequent_seqs(self) -> Union[torch.Tensor, None]:
        pass

    @abstractmethod
    def tp_frequent_seq_occurrences(self) -> Union[torch.Tensor, None]:
        pass

    @abstractmethod
    def tp_sequence_significance(self, source: SequenceSignificanceSource) -> Optional[torch.Tensor]:
        pass

    @abstractmethod
    def _learn_process_with_expert_idx(self) -> (SPFlockLearning, int):
        pass

    def cluster_center_projections_properties(self) -> List[ObserverPropertiesItem]:
        return self._cluster_projections_group_properties.get_properties()

    def cluster_center_projections(self, group_id: int) -> List[str]:
        result = []
        for tensor in self._get_inverse_projections(group_id):
            projected_tensor, _ = self._cluster_projections_group_properties.project_and_scale(tensor)
            result.append('data:image/png;base64,' + encode_image(projected_tensor))
        return result

    def _get_inverse_projections(self, projection_id: int) -> List[torch.Tensor]:
        all_projections = get_inverse_projections_for_all_clusters(self._node, self._expert_id)
        return all_projections[projection_id]


class TensorProviderExpertFlock(TensorProvider):
    def __init__(self, flock_node: 'ExpertFlockNode', expert_id: int, is_convolutional: bool = False):
        super().__init__(flock_node, expert_id)
        self.flock_node = flock_node
        self.expert_id = expert_id
        self.is_convolutional = is_convolutional

    def has_temporal_pooler(self) -> bool:
        return True

    def n_cluster_centers(self) -> int:
        return self.flock_node.params.n_cluster_centers

    def n_sequences(self) -> int:
        return self.flock_node.params.temporal.n_frequent_seqs

    def sequence_length(self) -> int:
        return self.flock_node.params.temporal.seq_length

    def current_cluster_id(self) -> int:
        tensor = self.flock_node.memory_blocks.sp.forward_clusters.tensor
        if not is_valid_tensor(tensor):
            return 0
        tensor = tensor._base if tensor._base is not None else tensor
        return tensor[self.expert_id].argmax().item()

    def _learn_process_with_expert_idx(self) -> (SPFlockLearning, int):
        if self.flock_node._unit is None:
            return None, 0

        process = self.flock_node._unit.flock.sp_flock.learn_process
        if process is None:
            return None, 0

        expert_id = 0 if self.is_convolutional else self.expert_id
        expert_mask = torch.eq(process.indices.squeeze(), expert_id)
        if not expert_mask.any():
            return None, 0
        matching_idx = expert_mask.argmax()
        return process, matching_idx

    def sp_cluster_centers(self):
        tensor = self.flock_node.memory_blocks.sp.cluster_centers.tensor
        return tensor[self.expert_id] if is_valid_tensor(tensor) else None

    def sp_execution_counter_learning(self):
        tensor = self.flock_node.memory_blocks.sp.execution_counter_learning.tensor
        return tensor[self.expert_id, 0] if is_valid_tensor(tensor) else None

    def tp_frequent_seqs(self):
        tensor = self.flock_node.memory_blocks.tp.frequent_seqs.tensor
        return tensor[self.expert_id] if is_valid_tensor(tensor) else None

    def tp_frequent_seq_occurrences(self):
        tensor = self.flock_node.memory_blocks.tp.frequent_seq_occurrences.tensor
        return tensor[self.expert_id] if is_valid_tensor(tensor) else None

    def tp_sequence_significance(self, source: SequenceSignificanceSource):
        if self.flock_node._unit is None:
            return None
        process = self.flock_node._unit.flock.tp_flock.trained_forward_process
        if process is None:
            return None

        if source == SequenceSignificanceSource.MODEL_FREQUENCY:
            return self.tp_frequent_seq_occurrences()

        if source == SequenceSignificanceSource.SEQ_LIKELIHOODS_ACTIVE:
            tensor_getter = lambda _: process.seq_likelihoods_active
        elif source == SequenceSignificanceSource.SEQ_LIKELIHOODS_CLUSTERS:
            tensor_getter = lambda _: process.seq_likelihoods_clusters
        elif source == SequenceSignificanceSource.SEQ_LIKELIHOODS_EXPLORATION:
            tensor_getter = lambda _: process.seq_likelihoods_exploration
        elif source == SequenceSignificanceSource.SEQ_LIKELIHOODS_GOAL_DIRECTED:
            tensor_getter = lambda _: process.seq_rewards_goal_directed
        elif source == SequenceSignificanceSource.SEQ_LIKELIHOODS_PRIORS_CLUSTERS:
            tensor_getter = lambda _: process.seq_likelihoods_priors_clusters
        elif source == SequenceSignificanceSource.SEQ_LIKELIHOODS_PRIORS_CLUSTERS_CONTEXT:
            tensor_getter = lambda _: process.seq_likelihoods_priors_clusters_context
        else:
            raise IllegalArgumentException(f'Unrecognized sequence significance source {source}')

        tensor = FlockProcessObservable(self.flock_node.params.flock_size, lambda: process, tensor_getter).get_tensor()
        return tensor[self.expert_id] if is_valid_tensor(tensor) else None


class TensorProviderSPFlock(TensorProvider):
    def __init__(self, sp_flock_node: 'SpatialPoolerFlockNode', expert_id: int, is_convolutional: bool = False):
        super().__init__(sp_flock_node, expert_id)
        self.sp_flock_node = sp_flock_node
        self.expert_id = expert_id
        self.is_convolutional = is_convolutional

    def has_temporal_pooler(self) -> bool:
        return False

    def n_cluster_centers(self) -> int:
        return self.sp_flock_node.params.n_cluster_centers

    def n_sequences(self) -> int:
        # No temporal pooler
        return 0

    def sequence_length(self) -> int:
        # No temporal pooler
        return 0

    def current_cluster_id(self) -> int:
        tensor = self.sp_flock_node.outputs.sp.forward_clusters.tensor
        if not is_valid_tensor(tensor):
            return 0
        tensor = tensor._base if tensor._base is not None else tensor
        return tensor[self.expert_id].argmax().item()

    def _learn_process_with_expert_idx(self) -> (SPFlockLearning, int):
        if self.sp_flock_node._unit is None:
            return None, 0

        process = self.sp_flock_node._unit.flock.learn_process
        if process is None:
            return None, 0

        expert_id = 0 if self.is_convolutional else self.expert_id
        expert_mask = torch.eq(process.indices.squeeze(), expert_id)
        if not expert_mask.any():
            return None, 0
        matching_idx = expert_mask.argmax()
        return process, matching_idx

    def sp_cluster_centers(self):
        tensor = self.sp_flock_node.memory_blocks.sp.cluster_centers.tensor

        if not is_valid_tensor(tensor):
            return None
        return tensor[self.expert_id]

    def sp_execution_counter_learning(self):
        tensor = self.sp_flock_node.memory_blocks.sp.execution_counter_learning.tensor
        return tensor[self.expert_id, 0] if is_valid_tensor(tensor) else None

    def tp_frequent_seqs(self):
        # No temporal pooler
        return None

    def tp_frequent_seq_occurrences(self):
        # No temporal pooler
        return None

    def tp_sequence_significance(self, source: SequenceSignificanceSource):
        # No temporal pooler
        return None


class ClusterObserver(ClusterObservable):
    _sequences_builder: SequencesBuilder
    _show_cluster_centers: bool
    _show_cluster_datapoints: bool
    _show_spring_lines: bool
    _show_spline_arrows: bool
    _projection_type: ClusterObserverProjection
    _prop_builder: ObserverPropertiesBuilder
    _n_cluster_centers: int
    _n_sequences: int
    _sequence_length: int

    _width: int = 640
    _height: int = 480
    _has_temporal_pooler: bool

    def __init__(self, tensor_provider: TensorProvider):
        self._has_temporal_pooler = tensor_provider.has_temporal_pooler()

        self._n_cluster_centers = tensor_provider.n_cluster_centers()
        self._n_sequences = tensor_provider.n_sequences()
        self._sequence_length = tensor_provider.sequence_length()

        self.cluster_centers = ClusterCentersDataBuilder(tensor_provider)
        self.fdsim = FDsimDataBuilder(tensor_provider)
        self.n_dims = 2
        self.pca = PcaDataBuilder(tensor_provider)
        self.spring_lines = SpringLinesBuilder(tensor_provider)
        self.spline_arrows = SplineArrowsBuilder(tensor_provider)
        self._prop_builder = ObserverPropertiesBuilder()
        self._sequences_builder = SequencesBuilder(tensor_provider)
        self._show_cluster_centers = True
        self._show_cluster_datapoints = True
        self._show_spring_lines = self._has_temporal_pooler
        self._show_spline_arrows = self._has_temporal_pooler
        self._projection_type = ClusterObserverProjection.PCA
        # self._pca_transformer = PcaTransformer()

    def get_data(self) -> ClusterObserverData:
        # if self._projection_type == ClusterObserverProjection.PCA:
        #     self.pca.update_pca_transformer(self._pca_transformer)

        return ClusterObserverData(
            cluster_centers=self.cluster_centers.get_data() if self._show_cluster_centers else None,
            fdsim=self.fdsim.get_data(),
            n_dims=self.n_dims,
            n_cluster_centers=self._n_cluster_centers,
            n_sequences=self._n_sequences,
            sequence_length=self._sequence_length,
            pca=self.pca.get_data(self.n_dims,
                                  self._show_cluster_datapoints) if self._projection_type == ClusterObserverProjection.PCA else None,
            projection_type="PCA" if self._projection_type == ClusterObserverProjection.PCA else "FDsim",
            width=self._width,
            height=self._height,
            spring_lines=self.spring_lines.get_data() if self._show_spring_lines else None,
            sequences=self._sequences_builder.get_data(),
            spline_arrows=self.spline_arrows.get_data() if self._show_spline_arrows else None,
        )

    def get_properties(self) -> List[ObserverPropertiesItem]:
        def update_projection_dim(value):
            if int(value) == 0:
                self.n_dims = 2
            else:
                self.n_dims = 3
            return value

        def update_show_cluster_centers(value: bool) -> bool:
            self._show_cluster_centers = value
            return value

        def update_show_cluster_datapoints(value: bool) -> bool:
            self._show_cluster_datapoints = value
            return value

        def update_show_spring_lines(value: bool) -> bool:
            self._show_spring_lines = value
            return value

        def update_show_spline_arrows(value: bool) -> bool:
            self._show_spline_arrows = value
            return value

        def format_projection_type(value: ClusterObserverProjection) -> int:
            if value == ClusterObserverProjection.PCA:
                return 0
            elif value == ClusterObserverProjection.FD_SIM:
                return 1
            else:
                raise IllegalArgumentException(f'Unrecognized projection {value}')

        def update_projection_type(value):
            old_type = self._projection_type
            if int(value) == 0:
                self._projection_type = ClusterObserverProjection.PCA
            elif int(value) == 1:
                self._projection_type = ClusterObserverProjection.FD_SIM
            else:
                raise IllegalArgumentException(f'Unrecognized projection {value}')

            if self._projection_type == ClusterObserverProjection.PCA and old_type != ClusterObserverProjection.PCA:
                self.pca.reset()

            return value

        def reset_projection(value):
            if self._projection_type == ClusterObserverProjection.PCA:
                self.pca.reset()
            elif self._projection_type == ClusterObserverProjection.FD_SIM:
                self.fdsim.reset()
            else:
                raise IllegalArgumentException(f'Unrecognized projection {value}')

        def update_width(value):
            self._width = int(value)
            return value

        def update_height(value):
            self._height = int(value)
            return value

        def yield_props():
            yield ObserverPropertiesItem('Projection', 'select', format_projection_type(self._projection_type),
                                         update_projection_type,
                                         select_values=[ObserverPropertiesItemSelectValueItem('PCA'),
                                                        ObserverPropertiesItemSelectValueItem(
                                                            'Force simulation')],
                                         state=ObserverPropertiesItemState.ENABLED if self._has_temporal_pooler else ObserverPropertiesItemState.READ_ONLY)

            yield ObserverPropertiesItem('Projection dimensionality', 'select', 0 if self.n_dims == 2 else 1,
                                         update_projection_dim, select_values=[
                    ObserverPropertiesItemSelectValueItem('2D'),
                    ObserverPropertiesItemSelectValueItem('3D')
                ])

            yield ObserverPropertiesItem('Reset Projection', 'button', "Reset", reset_projection)

            # Enablers
            yield self._prop_builder.checkbox('Show Cluster Centers', self._show_cluster_centers,
                                              update_show_cluster_centers)
            yield self._prop_builder.checkbox('Show Cluster Datapoints',
                                              self._show_cluster_datapoints if self._projection_type == ClusterObserverProjection.PCA else False,
                                              update_show_cluster_datapoints,
                                              state=ObserverPropertiesItemState.ENABLED if self._projection_type == ClusterObserverProjection.PCA else ObserverPropertiesItemState.DISABLED)
            yield self._prop_builder.checkbox('Show Spring Lines',
                                              self._show_spring_lines if self._has_temporal_pooler else False,
                                              update_show_spring_lines,
                                              state=ObserverPropertiesItemState.ENABLED if self._has_temporal_pooler else ObserverPropertiesItemState.DISABLED)
            yield self._prop_builder.checkbox('Show Spline Arrows',
                                              self._show_spline_arrows if self._has_temporal_pooler else False,
                                              update_show_spline_arrows,
                                              state=ObserverPropertiesItemState.ENABLED if self._has_temporal_pooler else ObserverPropertiesItemState.DISABLED)

            # Cluster Centers
            yield self._prop_builder.collapsible_header('Cluster Centers', default_is_expanded=True)
            yield from self.cluster_centers.get_properties(enabled=self._show_cluster_centers)

            # Spline Arrows
            yield self._prop_builder.collapsible_header('Spline Arrows', default_is_expanded=True)
            yield from self.spline_arrows.get_properties(enabled=self._show_spline_arrows)

            # Canvas
            yield self._prop_builder.collapsible_header('Canvas', default_is_expanded=True)
            yield ObserverPropertiesItem('Width', 'number', self._width, update_width)
            yield ObserverPropertiesItem('Height', 'number', self._height, update_height)

            # Force Simulation
            if self._has_temporal_pooler:
                yield ObserverPropertiesItem('Force simulation', 'collapsible_header', True, lambda _: "True")
                yield from self.fdsim.get_properties()

        return list(yield_props())


class ClusterObserverExpertFlock(ClusterObserver):

    def __init__(self, flock_node: 'ExpertFlockNode', expert_id: int, is_convolutional: bool = False):
        super().__init__(TensorProviderExpertFlock(flock_node, expert_id, is_convolutional))


class ClusterObserverSPFlock(ClusterObserver):

    def __init__(self, sp_flock_node: 'SpatialPoolerFlockNode', expert_id: int, is_convolutional: bool = False):
        super().__init__(TensorProviderSPFlock(sp_flock_node, expert_id, is_convolutional))


class ClusterUtils:

    @staticmethod
    def compute_similarities(cluster_centers_count: int, sequences: torch.Tensor,
                             sequence_occurrences: torch.Tensor) -> torch.Tensor:
        """Compute cluster centers similarities.

        Args:
            cluster_centers_count: number of cluster centers
            sequences: tensor[sequences_count, sequence_length] = cluster_center_id
            sequence_occurrences: tensor[sequences_count] = number_of_occurrences

        Returns:
            tensor[cluster_centers_count, cluster_centers_count] = similarity_of_given_cluster_centers
        """
        similarities = sequences.new_zeros((cluster_centers_count, cluster_centers_count), dtype=torch.float)
        occurred_seqs = sequence_occurrences > 0

        if not occurred_seqs.any():
            return similarities

        sequences = sequences[occurred_seqs]
        sequence_occurrences = sequence_occurrences[occurred_seqs]

        similarities_flat = similarities.view(-1)
        sequence_length = sequences.shape[1]
        for i in range(1, sequence_length):
            flat_indices = torch.add(sequences[:, i], cluster_centers_count, sequences[:, i - 1])
            similarities_flat.index_add_(0, flat_indices, sequence_occurrences)  # handles even repeated indices

        max_similarity = similarities.max()
        if max_similarity > 0:
            similarities /= max_similarity

        return similarities

    @staticmethod
    def compute_similarities_orderless(cluster_centers_count: int, sequences: torch.Tensor,
                                       sequence_occurrences: torch.Tensor) -> torch.Tensor:
        """Compute cluster centers similarities, disregarding order of cluster centers in sequences.

        Args:
            cluster_centers_count: number of cluster centers
            sequences: tensor[sequences_count, sequence_length] = cluster_center_id
            sequence_occurrences: tensor[sequences_count] = number_of_occurrences

        Returns:
            tensor[cluster_centers_count, cluster_centers_count] = similarity_of_given_cluster_centers
        """
        sequences_count = sequences.shape[0]
        masks = sequences.new_empty(size=(cluster_centers_count, sequences_count), dtype=torch.uint8)

        for c in range(cluster_centers_count):
            torch.any(sequences == c, dim=1, out=masks[c])

        masks = masks.float()
        similarities = torch.einsum('ij,kj,j->ik', masks, masks, sequence_occurrences)
        max_similarity = similarities.max()

        if max_similarity > 0:
            similarities /= max_similarity

        return similarities
