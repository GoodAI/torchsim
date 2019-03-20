from typing import List, Sequence
import logging

from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams, SamplingMethod
from torchsim.core.nodes import ExpertFlockNode, UnsqueezeNode, ForkNode, JoinNode, SpatialPoolerFlockNode
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.task0_base_topology import Task0BaseTopology

logger = logging.getLogger(__name__)


class Task0NarrowTopology(Task0BaseTopology):

    def __init__(self,
                 use_dataset: bool = True,
                 model_seed: int = 321,
                 baseline_seed: int = 333,
                 num_cc: Sequence[int] = (400, 300, 250),
                 batch_s: Sequence[int] = (1000, 1000, 700),
                 buffer_s: Sequence[int] = (3000, 3000, 3000),
                 sampling_m: SamplingMethod = SamplingMethod.LAST_N,
                 cbt: Sequence[int] = (1000, 1000, 1000),
                 lr: Sequence[float] = (0.3, 0.3, 0.3),
                 mbt: int = 1000,
                 class_filter=None,
                 label_scale: int = 1,
                 seq_len: int = 3):
        super().__init__(num_layers=3,
                         class_filter=class_filter,
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

        # TODO put in the static method
        for params in self._params_list:
            params.temporal.seq_length = seq_len

        # install experts
        self._install_experts(self._params_list, model_seed)
        self._connect_expert_output()
        self._install_baselines(self.flock_nodes, baseline_seed)

    def _install_experts(self, flock_params: List[ExpertParams], model_seed: int):

        self.flock_nodes = [ExpertFlockNode(flock_params[0], seed=model_seed),
                            ExpertFlockNode(flock_params[1], seed=model_seed),
                            SpatialPoolerFlockNode(flock_params[2], seed=model_seed)]

        self._top_level_flock_node = self.flock_nodes[-1]

        for node in self.flock_nodes:
            self.add_node(node)

        self._join_node = JoinNode(flatten=True)
        self.add_node(self._join_node)

        self._install_rescale(self._label_scale)

        unsqueeze_node = UnsqueezeNode(0)
        self.add_node(unsqueeze_node)

        # image -> unsqueeze_node -> SP1 -> SP2 -> join
        Connector.connect(
            self.se_io.outputs.image_output,
            unsqueeze_node.inputs.input)
        Connector.connect(
            unsqueeze_node.outputs.output,
            self.flock_nodes[0].inputs.sp.data_input)
        Connector.connect(
            self.flock_nodes[0].outputs.tp.projection_outputs,
            self.flock_nodes[1].inputs.sp.data_input)
        Connector.connect(
            self.flock_nodes[1].outputs.tp.projection_outputs,
            self._join_node.inputs[0])

        # label -> rescale ----> join
        Connector.connect(
            self.se_io.outputs.task_to_agent_label,
            self._rescale_node.inputs[0])
        Connector.connect(
            self._rescale_node.outputs[0],
            self._join_node.inputs[1])

        unsqueeze_node_2 = UnsqueezeNode(0)
        self.add_node(unsqueeze_node_2)

        # join -> unsqueeze_node -> top_level_expert
        Connector.connect(
            self._join_node.outputs.output,
            unsqueeze_node_2.inputs.input)
        Connector.connect(
            unsqueeze_node_2.outputs.output,
            self._top_level_flock_node.inputs.sp.data_input)

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
