from typing import List, Sequence
import logging

import torch

from eval_utils import run_just_model
from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams, SamplingMethod
from torchsim.core.nodes import ConvExpertFlockNode, UnsqueezeNode, JoinNode, ReceptiveFieldNode, SpatialPoolerFlockNode
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.task0_base_topology import Task0BaseTopology
from torchsim.utils.param_utils import Size2D

logger = logging.getLogger(__name__)


class ConvWideTwoLayerTopology(Task0BaseTopology):
    _experts_on_x: int

    _lrf_input_dims: torch.Size
    # _parent_rf_dims: tuple

    _node_lrf: ReceptiveFieldNode

    NUM_LAYERS = 2

    def __init__(self,
                 use_dataset: bool = True,
                 image_size=SeDatasetSize.SIZE_24,  # small one
                 model_seed: int = 321,
                 baseline_seed: int = 333,
                 num_cc: Sequence[int] = (150, 300),
                 batch_s: Sequence[int] = (3000, 1000),
                 buffer_s: Sequence[int] = (6000, 6000),
                 sampling_m: SamplingMethod = SamplingMethod.BALANCED,
                 cbt: Sequence[int] = (1000, 1000),
                 lr=(0.2, 0.2),
                 mbt: int = 1000,
                 class_filter=None,
                 experts_on_x: int = 2,
                 label_scale: float = 1,
                 seq_len: int = 3):
        super().__init__(num_layers=self.NUM_LAYERS,
                         class_filter=class_filter,
                         image_size=image_size,
                         use_dataset=use_dataset,
                         label_scale=label_scale)

        # create list of flock params
        self._params_list = self.create_flock_params(
            num_cluster_centers=num_cc,
            learning_rate=lr,
            batch_size=batch_s,
            buffer_size=buffer_s,
            sampling_method=sampling_m,
            cluster_boost_threshold=cbt,
            flock_size=1,
            max_boost_time=mbt,
            num_layers=self._num_layers
        )
        # TODO move to the method above
        for param in self._params_list:
            param.temporal.seq_length = seq_len

        self._label_scale = label_scale
        self._experts_on_x = experts_on_x
        bottom_flock_size = self._experts_on_x ** 2
        self._change_bottom_expert(self._params_list, flock_size=bottom_flock_size)

        # install experts
        self._install_experts(self._params_list, model_seed)
        self._connect_expert_output()
        self._install_baselines(self.flock_nodes, baseline_seed)

    @staticmethod
    def _change_bottom_expert(params_list: List[ExpertParams], flock_size: int):
        params = params_list[0]
        params.flock_size = flock_size

    def _install_lrf(self, image_size: int, experts_on_x: int):
        self._lrf_input_dims = torch.Size((image_size, image_size, 3))
        self._parent_rf_dims = Size2D(image_size // experts_on_x, image_size // experts_on_x)

        self._node_lrf = ReceptiveFieldNode(input_dims=self._lrf_input_dims, parent_rf_dims=self._parent_rf_dims)
        self.add_node(self._node_lrf)

    def _install_experts(self, flock_params: List[ExpertParams], model_seed: int):

        self.flock_nodes = [ConvExpertFlockNode(flock_params[0], seed=model_seed),
                            SpatialPoolerFlockNode(flock_params[1], seed=model_seed)]

        self._top_level_flock_node = self.flock_nodes[-1]

        for node in self.flock_nodes:
            self.add_node(node)

        # lrf
        self._install_lrf(self._image_size.value, self._experts_on_x)

        # join output of expert and label
        self._join_node = JoinNode(flatten=True)
        self.add_node(self._join_node)

        # scale node
        self._install_rescale(self._label_scale)

        # image -> LRF -> E1 -> join
        Connector.connect(
            self.se_io.outputs.image_output,
            self._node_lrf.inputs.input)
        Connector.connect(
            self._node_lrf.outputs.output,
            self.flock_nodes[0].inputs.sp.data_input)
        Connector.connect(
            self.flock_nodes[0].outputs.tp.projection_outputs,
            self._join_node.inputs[0])

        # label -> rescale ---> join
        Connector.connect(
            self.se_io.outputs.task_to_agent_label,
            self._rescale_node.inputs[0])
        Connector.connect(
            self._rescale_node.outputs[0],
            self._join_node.inputs[1])

        self.unsqueeze_node = UnsqueezeNode(0)
        self.add_node(self.unsqueeze_node)

        # join -> unsqueeze_node -> top_level_expert
        Connector.connect(
            self._join_node.outputs.output,
            self.unsqueeze_node.inputs.input)
        Connector.connect(
            self.unsqueeze_node.outputs.output,
            self._top_level_flock_node.inputs.sp.data_input)

    def restart(self):
        pass


if __name__ == '__main__':
    params = [
        {'num_cc': [150, 300],
         'lr': [0.1, 0.001],  # 0.2 0.2 - learns reasonably fast (oscillating)
         'batch_s': [3000, 1000],
         'label_scale': 1}
    ]

    run_just_model(ConvWideTwoLayerTopology(**params[0]), gui=True)
