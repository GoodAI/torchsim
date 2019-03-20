from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.models.flock.expert_flock import ConvExpertFlock
from torchsim.core.nodes.expert_node import ExpertFlockNode, ExpertFlockUnit, ExpertFlockInternals, ExpertFlockOutputs, \
    ExpertFlockInputs
from torchsim.gui.observers.cluster_observer import ClusterObserverExpertFlock
from torchsim.utils.seed_utils import set_global_seeds


class ConvExpertFlockUnit(ExpertFlockUnit):
    @staticmethod
    def _create_flock(params, creator):
        return ConvExpertFlock(params, creator)


class ConvExpertFlockInternals(ExpertFlockInternals):
    def __init__(self, owner):
        super().__init__(owner)
        # SP class observables
        self.sp_common_cluster_centers = self.create("SP_common_cluster_centers")

        # Buffer observables
        self.sp_common_buffer_inputs = self.create_buffer("SP_common_buffer_inputs")
        self.sp_common_buffer_clusters = self.create_buffer("SP_common_buffer_clusters")

    def prepare_slots(self, unit: ConvExpertFlockUnit):
        super().prepare_slots(unit)
        self.sp_common_cluster_centers.tensor = unit.flock.sp_flock.common_cluster_centers

        # Buffer observables
        self.sp_common_buffer_inputs.buffer = unit.flock.sp_flock.common_buffer.inputs
        self.sp_common_buffer_clusters.buffer = unit.flock.sp_flock.common_buffer.clusters

        # Reshape some of the tensors to match the input space.
        receptive_field_shape = self._owner.get_receptive_field_shape()

        # SP Observables
        self.sp_common_cluster_centers.reshape_tensor(receptive_field_shape)

        # SP Buffer
        self.sp_common_buffer_inputs.reshape_tensor(receptive_field_shape)


class ConvExpertFlockNode(ExpertFlockNode):
    """Convolutional version of the ExpertFlock.

    Currently, only the spatial pooler is implemented as convolutional, the temporal experts are independent as in the
    ExpertFlockNode. All memory blocks should be observable as in the non-convolutional version, plus there are
    additional memory blocks starting with `common_` related to the convolutional mechanics.
    """

    def __init__(self, params: ExpertParams, seed: int = None, name="Conv ExpertFlock"):
        super().__init__(params, seed, name)

    def get_memory_blocks(self):
        return ExpertFlockInputs(self), ConvExpertFlockInternals(self), ExpertFlockOutputs(self)

    def _create_unit(self, creator: TensorCreator):
        self._derive_params()
        set_global_seeds(self._seed)
        return ConvExpertFlockUnit(creator, self.params)

    def _get_sp_learn_flock_size(self):
        """What is the real flock size when learning (flock_size or 1 in conv SP)."""

        return 1

    def _create_cluster_observer(self, number):
        return ClusterObserverExpertFlock(self, number, is_convolutional=True)
