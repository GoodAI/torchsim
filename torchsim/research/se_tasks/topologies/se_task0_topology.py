from abc import abstractmethod
from typing import List

from torchsim.core.eval.node_accessors.se_io_accessor import SeIoAccessor
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes.dataset_se_objects_node import DatasetConfig, DatasetSeObjectsParams
from torchsim.core.nodes.fork_node import ForkNode
from torchsim.core.nodes.join_node import JoinNode
from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode
from torchsim.research.se_tasks.topologies.se_io.se_io_base import SeIoBase
from torchsim.research.se_tasks.topologies.se_io.se_io_task0 import SeIoTask0
from torchsim.research.se_tasks.topologies.se_io.se_io_task0_dataset import SeIoTask0Dataset
from torchsim.research.se_tasks.topologies.se_task_topology import TestableTopology


class SeT0TopologicalGraph(TestableTopology):
    """Base class for SE Task 0 topologies."""
    se_io: SeIoBase  # installer: a thing which deals with the dataset and SE in a common way

    _join_node: JoinNode
    _top_level_flock_node: SpatialPoolerFlockNode
    _fork_node: ForkNode

    def __init__(self, curriculum: tuple = (0, -1), use_dataset: bool = False, save_gpu_memory: bool = False,
                 class_filter: List[int] = None, location_filter: float = 1.0, run_init=True):
        super().__init__()

        if run_init:  # a small hack to allow to postpone init until children have set their parameters
            self.create_se_io(curriculum, use_dataset, save_gpu_memory, class_filter, location_filter)
            self.init()

    def create_se_io(self, curriculum: tuple, use_dataset: bool, save_gpu_memory: bool, class_filter: List[int],
                     location_filter: float):
        self.se_io = SeT0TopologicalGraph._get_installer(use_dataset, curriculum, save_gpu_memory, class_filter,
                                                         location_filter)

    def init(self):
        self.se_io.install_nodes(self)
        self._join_node = JoinNode(flatten=True)
        self.add_node(self._join_node)
        self._install_experts()
        self._connect_expert_output()

    @staticmethod
    def _get_installer(use_dataset: bool, curriculum: tuple, save_gpu_memory: bool, class_filter: List[int],
                       location_filter: float):
        if use_dataset:
            return SeIoTask0Dataset(
                DatasetSeObjectsParams(dataset_config=DatasetConfig.TRAIN_ONLY,
                                       save_gpu_memory=save_gpu_memory,
                                       class_filter=class_filter,
                                       location_filter_ratio=location_filter))
        else:
            return SeIoTask0(curriculum)

    @abstractmethod
    def _install_experts(self):
        pass

    def _connect_expert_output(self):
        label_size = SeIoAccessor.get_num_labels(self.se_io)

        self._fork_node = ForkNode(1, [self._top_level_expert_output_size(), label_size])
        self.add_node(self._fork_node)
        Connector.connect(self._get_agent_output(),
                          self._fork_node.inputs.input)
        Connector.connect(self._fork_node.outputs[1], self.se_io.inputs.agent_to_task_label,
                          is_backward=True)

    @abstractmethod
    def _get_agent_output(self) -> MemoryBlock:
        pass

    @abstractmethod
    def _top_level_expert_output_size(self):
        pass

    @staticmethod
    def _create_expert_params() -> ExpertParams:
        expert_params = ExpertParams()
        expert_params.flock_size = 1
        expert_params.n_cluster_centers = 20
        expert_params.compute_reconstruction = True
        return expert_params

    def is_in_testing_phase(self):
        return self.se_io.get_testing_phase() == 1
