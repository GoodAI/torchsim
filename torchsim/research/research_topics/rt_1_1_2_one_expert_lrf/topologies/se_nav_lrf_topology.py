import logging
import torch

from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.models.receptive_field.grid import Stride
from torchsim.core.nodes import DatasetSeNavigationNode, DatasetSENavigationParams, SamplingMethod
from torchsim.core.nodes import ReceptiveFieldNode
from torchsim.core.nodes import SpatialPoolerFlockNode
from torchsim.utils.param_utils import Size2D

logger = logging.getLogger(__name__)


class SeNavLrfTopology(Topology):
    def __init__(self,
                 expert_width: int = 1, input_square_side: int = 64,
                 n_cluster_centers: int = 8,
                 stride: int = None,
                 training_phase_steps: int = 200,
                 testing_phase_steps: int = 800,
                 seed: int = 0
                 ):
        super().__init__(device='cuda')

        if stride is None:
            stride = expert_width

        self.training_phase_steps = training_phase_steps
        self.testing_phase_steps = testing_phase_steps
        self.testing_phase = True
        self.n_testing_phase = -1
        self.training_step = -1
        self._step_counter = 0
        assert (input_square_side - (expert_width - stride)) % stride == 0, \
            f'(input_square_side - (expert_width - stride)) ' \
                f'({(input_square_side - (expert_width - stride))}) must be divisible' \
                f' by stride ({stride})'
        self.n_experts_width = (input_square_side - (expert_width - stride)) // stride
        self.input_square_side = input_square_side
        self.one_expert_lrf_width = expert_width

        self.sp_params = ExpertParams()
        self.sp_params.n_cluster_centers = n_cluster_centers
        self.sp_params.spatial.input_size = self.one_expert_lrf_width * self.one_expert_lrf_width * 3
        self.sp_params.flock_size = int(self.n_experts_width ** 2)
        self.sp_params.spatial.buffer_size = 510
        self.sp_params.spatial.batch_size = 500
        self.sp_params.compute_reconstruction = True

        self.reconstructed_data = torch.zeros((self.input_square_side, self.input_square_side, 3),
                                              dtype=self.float_dtype,
                                              device=self.device)
        self.image_difference = torch.zeros((self.input_square_side, self.input_square_side, 3), dtype=self.float_dtype,
                                            device=self.device)

        se_nav_params = DatasetSENavigationParams(SeDatasetSize.SIZE_64,
                                                  sampling_method=SamplingMethod.RANDOM_SAMPLING)

        self._node_se_nav = DatasetSeNavigationNode(se_nav_params, seed=0)

        parent_rf_dims = Size2D(self.one_expert_lrf_width, self.one_expert_lrf_width)
        self._node_lrf = ReceptiveFieldNode((input_square_side, input_square_side, 3), parent_rf_dims,
                                            Stride(stride, stride))

        # necessary to clone the params, because the GUI changes them during the simulation (restart needed)
        self._node_spatial_pooler = SpatialPoolerFlockNode(self.sp_params.clone(), seed=seed)
        self._node_spatial_pooler_backup = SpatialPoolerFlockNode(self.sp_params.clone(), seed=seed)

        self.add_node(self._node_se_nav)
        self.add_node(self._node_lrf)
        self.add_node(self._node_spatial_pooler)
        self.add_node(self._node_spatial_pooler_backup)

        Connector.connect(
            self._node_se_nav.outputs.image_output,
            self._node_lrf.inputs.input
        )
        Connector.connect(
            self._node_lrf.outputs.output,
            self._node_spatial_pooler.inputs.sp.data_input
        )
        Connector.connect(
            self._node_lrf.outputs.output,
            self._node_spatial_pooler_backup.inputs.sp.data_input
        )

        self.se_nav_node = self._node_se_nav

    def step(self):
        if not self._is_initialized:
            self._assign_ids_to_nodes(self._id_generator)
            self.order_nodes()
            self._update_memory_blocks()
            self.remove_node(self._node_spatial_pooler_backup)
        #     # self.remove_node(self._node_mnist_test)

        if not self.testing_phase and self._step_counter % self.training_phase_steps == 0:
            self.testing_phase = True
            self.n_testing_phase += 1
            self.set_testing_model()
            self._step_counter = 0
        elif self.testing_phase and self._step_counter % self.testing_phase_steps == 0:
            self.testing_phase = False
            self.set_training_model()
            self._step_counter = 0
        self._step_counter += 1

        if not self.testing_phase:
            self.training_step += 1

        if not self._is_initialized:
            self._assign_ids_to_nodes(self._id_generator)
            self.order_nodes()
            self._is_initialized = True

        super().step()

        flock_reconstruction = self._node_spatial_pooler.outputs.sp.current_reconstructed_input
        # noinspection PyProtectedMember
        self.reconstructed_data.copy_(self._node_lrf.inverse_projection(flock_reconstruction.tensor))

        self.image_difference.copy_(self.reconstructed_data - self._node_se_nav.outputs[0].tensor)

    def set_training_model(self):
        # noinspection PyProtectedMember
        self._node_spatial_pooler_backup._unit.copy_to(self._node_spatial_pooler._unit)
        self._node_spatial_pooler.switch_learning(True)

    def set_testing_model(self):
        # noinspection PyProtectedMember
        self._node_spatial_pooler._unit.copy_to(self._node_spatial_pooler_backup._unit)
        self._node_spatial_pooler.switch_learning(False)
