from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.spatial_pooler import ConvSPFlock, ExpertParams
from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode, SPFlockInternals, SPFlockInputs, SPFlockOutputs, \
    SpatialPoolerFlockUnit
from torchsim.gui.observers.cluster_observer import ClusterObserverSPFlock
from torchsim.utils.seed_utils import set_global_seeds


class ConvSpatialPoolerFlockUnit(SpatialPoolerFlockUnit):

    @staticmethod
    def _create_flock(params, creator):
        return ConvSPFlock(params, creator)


class ConvSPFlockInternals(SPFlockInternals):
    def __init__(self, owner):
        super().__init__(owner)
        # SP class observables
        self.sp_common_cluster_centers = self.create("SP_common_cluster_centers")

        # Buffer observables
        self.sp_common_buffer_inputs = self.create_buffer("SP_common_buffer_inputs")
        self.sp_common_buffer_clusters = self.create_buffer("SP_common_buffer_clusters")

    def prepare_slots(self, unit: ConvSpatialPoolerFlockUnit):
        super().prepare_slots(unit)
        self.sp_common_cluster_centers.tensor = unit.flock.common_cluster_centers

        # Buffer observables
        self.sp_common_buffer_inputs.buffer = unit.flock.common_buffer.inputs
        self.sp_common_buffer_clusters.buffer = unit.flock.common_buffer.clusters

        # TODO (Refactor): DRY this (see expert_node).
        # Reshape some of the tensors to match the input space.
        receptive_field_shape = self._owner.get_receptive_field_shape()

        # SP Observables
        self.sp_common_cluster_centers.reshape_tensor(receptive_field_shape)

        # SP Buffer
        self.sp_common_buffer_inputs.reshape_tensor(receptive_field_shape)


class ConvSpatialPoolerFlockNode(SpatialPoolerFlockNode):
    """Convolutional version of the SpatialPoolerFlockNode.

    All memory blocks should be observable as in the non-convolutional version, plus there are
    additional memory blocks starting with `common_` related to the convolutional mechanics.
    """

    def __init__(self, params: ExpertParams, seed: int = None, name="Conv SPFlock"):
        super().__init__(params, seed, name)

    def get_memory_blocks(self):
        return SPFlockInputs(self), ConvSPFlockInternals(self), SPFlockOutputs(self)

    def _create_unit(self, creator: TensorCreator) -> ConvSpatialPoolerFlockUnit:
        self._derive_params()
        set_global_seeds(self._seed)
        return ConvSpatialPoolerFlockUnit(creator, self.params)

    def _get_sp_learn_flock_size(self):
        """What is the real flock size when learning (flock_size or 1 in conv SP)."""

        return 1

    def _create_cluster_observer(self, number):
        return ClusterObserverSPFlock(self, number, is_convolutional=True)
