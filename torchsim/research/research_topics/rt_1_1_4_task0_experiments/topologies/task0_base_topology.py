from abc import ABC
from typing import List, Sequence
import logging

from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.eval.node_accessors.flock_node_accessor import FlockNodeAccessor
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams, SamplingMethod
from torchsim.core.nodes import ForkNode
from torchsim.core.nodes import DatasetSeObjectsParams, DatasetConfig
from torchsim.core.nodes import ExpertFlockNode
from torchsim.core.nodes import JoinNode
from torchsim.core.nodes import LambdaNode
from torchsim.core.nodes import RandomNumberNode
from torchsim.core.nodes import SpatialPoolerFlockNode
from torchsim.core.nodes import ConstantNode
from torchsim.research.se_tasks.topologies.se_io.se_io_base import SeIoBase
from torchsim.research.se_tasks.topologies.se_io.se_io_task0 import SeIoTask0
from torchsim.research.se_tasks.topologies.se_io.se_io_task0_dataset import SeIoTask0Dataset

logger = logging.getLogger(__name__)


class Task0BaseTopology(Topology, ABC):
    """Base class for SE Task 0 topologies."""

    se_io: SeIoBase
    _join_node: JoinNode
    fork_node: ForkNode
    _rescale_node: LambdaNode

    _top_level_flock_node: SpatialPoolerFlockNode

    # baselines for each layer separately (with corresponding output size) and for class label predictions
    baselines: List[RandomNumberNode]
    label_baseline: ConstantNode
    random_label_baseline: RandomNumberNode

    device: str = 'cuda'

    _num_layers: int

    flock_nodes: List[ExpertFlockNode]
    _params_list: List[ExpertParams]
    current_step: int
    _is_learning: bool

    _label_scale: float
    _image_size: SeDatasetSize

    def __init__(self,
                 num_layers: int=3,
                 use_dataset: bool = True,
                 class_filter=None,
                 label_scale: float=1,
                 image_size: SeDatasetSize = SeDatasetSize.SIZE_64):
        super().__init__(self.device)

        self._num_layers = num_layers  # TODO not parametrized for now
        self._label_scale = label_scale
        self.current_step = 0
        self._is_learning = False
        self._image_size = image_size

        # install the dataset
        self.se_io = self._get_installer(use_dataset,
                                         self.get_dataset_params(class_filter=class_filter, size=image_size))
        self.se_io.install_nodes(self)

    @staticmethod
    def get_dataset_params(
            class_filter: List[int] = None,
            size: SeDatasetSize=SeDatasetSize.SIZE_64):

        params = DatasetSeObjectsParams()
        params.dataset_config = DatasetConfig.TRAIN_ONLY
        params.dataset_size = size
        params.class_filter = class_filter
        return params

    def get_image_size(self) -> int:
        return self._image_size.value

    @staticmethod
    def create_flock_params(num_cluster_centers: Sequence[int],
                            learning_rate: Sequence[float],
                            buffer_size: Sequence[int],
                            batch_size: Sequence[int],
                            sampling_method: SamplingMethod,
                            cluster_boost_threshold: Sequence[int],
                            flock_size: int,
                            max_boost_time: int,
                            num_layers: int):

        params_list = []

        assert len(num_cluster_centers) == num_layers
        assert len(buffer_size) == num_layers
        assert len(batch_size) == num_layers
        assert len(learning_rate) == num_layers

        for layer_id in range(0, num_layers):
            params = ExpertParams()

            params.n_cluster_centers = num_cluster_centers[layer_id]
            params.flock_size = flock_size

            params.spatial.buffer_size = buffer_size[layer_id]
            params.spatial.batch_size = batch_size[layer_id]
            params.spatial.cluster_boost_threshold = cluster_boost_threshold[layer_id]
            params.spatial.max_boost_time = max_boost_time  # should be bigger than any cluster_boost_threshold
            params.spatial.learning_rate = learning_rate[layer_id]
            params.spatial.sampling_method = sampling_method
            # params.compute_reconstruction = True

            params_list.append(params)

        # just the top expert computes reconstruction
        params_list[-1].compute_reconstruction = True

        return params_list

    def get_num_layers(self):
        return self._num_layers

    @staticmethod
    def _get_installer(use_dataset: bool, dataset_params=None):
        if use_dataset:
            return SeIoTask0Dataset(dataset_params)  # default params used here
        else:
            # TODO proper configuration of SE not passed here (curriculum etc..)
            return SeIoTask0()

    def _install_baselines(self, flock_layers: List[ExpertFlockNode], baseline_seed: int):
        """For each layer in the topology installs own random baseline with a corresponding output size."""
        self.baselines = []
        for layer in flock_layers:
            output_dimension = FlockNodeAccessor.get_sp_output_size(layer)

            node = RandomNumberNode(upper_bound=output_dimension, seed=baseline_seed)
            self.add_node(node)
            self.baselines.append(node)

        # baseline for the labels separately
        self.label_baseline = ConstantNode(shape=self.se_io.get_num_labels(), constant=0, name='label_const')
        self.random_label_baseline = RandomNumberNode(upper_bound=self.se_io.get_num_labels(), seed=baseline_seed)

        self.add_node(self.label_baseline)
        self.add_node(self.random_label_baseline)

    def _install_rescale(self, scale: float):
        # rescale label - bigger weight
        def rescale(inputs, outputs):
            outputs[0].copy_(inputs[0] * scale)

        self._rescale_node = LambdaNode(rescale, 1, [(20,)])  # TODO constant here
        self.add_node(self._rescale_node)

    @staticmethod
    def _almost_top_level_expert_output_size(param_list: List[ExpertParams]):
        """For the fork dimensions."""
        params = param_list[-2]
        return params.n_cluster_centers * params.flock_size

    def _connect_expert_output(self):

        self.fork_node = ForkNode(1,
                                  [self._almost_top_level_expert_output_size(self._params_list),
                                    self.se_io.get_num_labels()
                                    ])
        self.add_node(self.fork_node)

        # top-level-expert -> fork
        Connector.connect(
            self._top_level_flock_node.outputs.sp.current_reconstructed_input,
            self.fork_node.inputs.input)
        # fork -> dataset/se
        Connector.connect(
            self.fork_node.outputs[1],
            self.se_io.inputs.agent_to_task_label,
            is_backward=True)

    def step(self):
        super().step()
        self.current_step += 1

    def switch_learning(self, learning_on: bool):
        logger.info(f'Topology: changing the learning state to learning_on: {learning_on}')

        try:
            for node in self.flock_nodes:
                node.switch_learning(learning_on)

            self._is_learning = learning_on
            logger.info(f'switching learning to {learning_on} at sim. step: {self.current_step}')
        except Exception:
            logger.info(f'unable to switch the learning to {learning_on} at sim. step: {self.current_step}')
            raise

    @property
    def is_learning(self):
        return self._is_learning
