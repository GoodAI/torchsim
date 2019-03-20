from typing import List, Sequence
import logging

from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams, SamplingMethod
from torchsim.core.nodes import ConvExpertFlockNode, UnsqueezeNode, ExpertFlockNode, JoinNode, SpatialPoolerFlockNode
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.task0_conv_wide_topology import \
    Task0ConvWideTopology

logger = logging.getLogger(__name__)


class Task0ConvWideTopologyMoreLabels(Task0ConvWideTopology):
    _mid_join_node: JoinNode

    def __init__(self,
                 use_dataset: bool = True,
                 image_size: SeDatasetSize = SeDatasetSize.SIZE_32,
                 model_seed: int = 321,
                 baseline_seed: int = 333,
                 num_cc: Sequence[int] = (150, 150, 150),
                 batch_s: Sequence[int] = (3000, 1000, 500),
                 buffer_s: Sequence[int] = (5000, 3000, 1000),
                 sampling_m: SamplingMethod = SamplingMethod.LAST_N,
                 cbt: Sequence[int] = (1000, 1000, 1000),
                 lr: Sequence[float] = (0.1, 0.1, 0.1),
                 mbt: int = 1000,
                 class_filter=None,
                 experts_on_x: int = 4,
                 label_scale: float = 1,
                 seq_len: int = 3):
        super().__init__(model_seed=model_seed,
                         image_size=image_size,
                         baseline_seed=baseline_seed,
                         use_dataset=use_dataset,
                         num_cc=num_cc,
                         batch_s=batch_s,
                         buffer_s=buffer_s,
                         sampling_m=sampling_m,
                         cbt=cbt,
                         lr=lr,
                         mbt=mbt,
                         class_filter=class_filter,
                         experts_on_x=experts_on_x,
                         label_scale=label_scale,
                         seq_len=seq_len)

    def _install_experts(self, flock_params: List[ExpertParams], model_seed: int):
        self._flock_nodes = [ConvExpertFlockNode(flock_params[0], seed=model_seed),
                             ExpertFlockNode(flock_params[1], seed=model_seed),
                             SpatialPoolerFlockNode(flock_params[2], seed=model_seed)]

        self._top_level_flock_node = self._flock_nodes[-1]

        for node in self._flock_nodes:
            self.add_node(node)

        # lrf
        self._install_lrf(self._image_size.value, self._experts_on_x)

        # join output of expert and label
        self._join_node = JoinNode(flatten=True)
        self.add_node(self._join_node)

        self._mid_join_node = JoinNode(flatten=True)
        self.add_node(self._mid_join_node)

        # scale node
        self._install_rescale(self._label_scale)

        unsqueeze_node_0 = UnsqueezeNode(0)
        self.add_node(unsqueeze_node_0)

        unsqueeze_node_1 = UnsqueezeNode(0)
        self.add_node(unsqueeze_node_1)

        # image -> LRF -> E1 -> mid_join -> E2 -> join
        Connector.connect(
            self.se_io.outputs.image_output,
            self._node_lrf.inputs.input)
        Connector.connect(
            self._node_lrf.outputs.output,
            self._flock_nodes[0].inputs.sp.data_input)
        Connector.connect(
            self._flock_nodes[0].outputs.tp.projection_outputs,
            self._mid_join_node.inputs[0])
        # self._flock_nodes[1].inputs.sp.data_input)
        Connector.connect(
            self._mid_join_node.outputs[0],
            unsqueeze_node_0.inputs.input)
        Connector.connect(
            unsqueeze_node_0.outputs.output,
            self._flock_nodes[1].inputs.sp.data_input)
        Connector.connect(
            self._flock_nodes[1].outputs.tp.projection_outputs,
            self._join_node.inputs[0])

        # label -> rescale --------> join
        Connector.connect(
            self.se_io.outputs.task_to_agent_label,
            self._rescale_node.inputs[0])
        Connector.connect(
            self._rescale_node.outputs[0],
            self._join_node.inputs[1])

        #         rescale --> mid join
        Connector.connect(
            self._rescale_node.outputs[0],
            self._mid_join_node.inputs[1])

        # join -> top_level_expert
        Connector.connect(
            self._join_node.outputs.output,
            unsqueeze_node_1.inputs.input)
        Connector.connect(
            unsqueeze_node_1.outputs.output,
            self._top_level_flock_node.inputs.sp.data_input)
