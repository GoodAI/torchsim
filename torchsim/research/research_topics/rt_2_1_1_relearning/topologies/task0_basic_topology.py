import logging
from typing import List

from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams, SamplingMethod
from torchsim.core.nodes import SpatialPoolerFlockNode, UnsqueezeNode
from torchsim.core.nodes.random_number_node import RandomNumberNode
from torchsim.core.nodes.constant_node import ConstantNode
from torchsim.research.se_tasks.topologies.se_task0_topology import SeT0TopologicalGraph


logger = logging.getLogger(__name__)


class SeT0BasicTopologyRT211(SeT0TopologicalGraph):
    """A model which receives data from the 0th SE task and learns spatial patterns."""
    LEARNING_RATE = 0.2
    _label_baseline: ConstantNode
    _random_label_baseline: RandomNumberNode

    def __init__(self, curriculum: tuple = (0, -1), use_dataset: bool = True, save_gpu_memory: bool=False,
                 class_filter: List[int] = None, location_filter: float = 1.0, num_ccs: int = 20,
                 buffer_size: int=1000, sampling_method: SamplingMethod = SamplingMethod.BALANCED, run_init=True):
        super().__init__(curriculum, use_dataset, save_gpu_memory, class_filter, location_filter, run_init=False)
        self._num_ccs = num_ccs
        self._buffer_size = buffer_size
        self._sampling_method = sampling_method
        self._current_step = 0
        if run_init:  # a small hack to allow to postpone init until children have set their parameters
            self.create_se_io(curriculum, use_dataset, save_gpu_memory, class_filter, location_filter)
            self.init()

        self._is_learning = True

    def init(self):
        super().init()
        self._install_baselines()

    def _install_baselines(self):
        self._label_baseline = ConstantNode(shape=self.se_io.get_num_labels(), constant=0, name='label_const')
        self._random_label_baseline = RandomNumberNode(upper_bound=self.se_io.get_num_labels(), seed=0)

        self.add_node(self._label_baseline)
        self.add_node(self._random_label_baseline)

    def _install_experts(self):
        self._top_level_flock_node = SpatialPoolerFlockNode(self._create_expert_params())
        self._flock_nodes = [self._top_level_flock_node]
        self.add_node(self._top_level_flock_node)
        self.unsqueeze_node = UnsqueezeNode(0)
        self.add_node(self.unsqueeze_node)
        Connector.connect(self.se_io.outputs.image_output, self._join_node.inputs[0])
        Connector.connect(self.se_io.outputs.task_to_agent_label, self._join_node.inputs[1])
        Connector.connect(self._join_node.outputs[0], self.unsqueeze_node.inputs.input)
        Connector.connect(self.unsqueeze_node.outputs.output, self._top_level_flock_node.inputs.sp.data_input)

    def _get_agent_output(self):
        return self._top_level_flock_node.outputs.sp.current_reconstructed_input

    def _top_level_expert_output_size(self):
        return self.se_io.get_image_numel()

    def _create_expert_params(self) -> ExpertParams:
        expert_params = ExpertParams()
        expert_params.flock_size = 1
        expert_params.n_cluster_centers = self._num_ccs
        expert_params.compute_reconstruction = True
        expert_params.spatial.batch_size = 990
        expert_params.spatial.buffer_size = self._buffer_size
        expert_params.spatial.cluster_boost_threshold = self._num_ccs*2
        expert_params.spatial.learning_rate = SeT0BasicTopologyRT211.LEARNING_RATE
        expert_params.spatial.sampling_method = self._sampling_method
        expert_params.spatial.learning_period = 10
        expert_params.spatial.max_boost_time = 5000
        return expert_params

    def before_step(self):
        super().before_step()
        self._current_step += 1

    def switch_learning(self, learning_on: bool):
        logger.info(f'Topology: changing the learning state to learning_on: {learning_on}')

        try:
            for node in self._flock_nodes:
                node.switch_learning(learning_on)

            self._is_learning = learning_on
            logger.warning(f'switching learning to {learning_on} at sim. step: {self._current_step}')
        except:
            logger.warning(f'unable to switch the learning to {learning_on} at sim. step: {self._current_step}')

    def is_in_testing_phase(self):
        return not self._is_learning
