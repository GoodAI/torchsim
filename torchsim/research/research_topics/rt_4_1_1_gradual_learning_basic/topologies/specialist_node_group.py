from typing import List

from functools import partial

import torch
import logging
from dataclasses import dataclass

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_base import NodeBase
from torchsim.core.graph.node_group import NodeGroupBase, GroupInputs, GroupOutputs
from torchsim.core.models.expert_params import SpatialPoolerParams, TemporalPoolerParams, ExpertParams
from torchsim.core.nodes import ExpertFlockNode, LambdaNode, ToOneHotNode
from torchsim.core.nodes.flatten_node import FlattenNode
from torchsim.core.nodes.to_one_hot_node import ToOneHotMode

logger = logging.getLogger(__name__)


@dataclass
class SpecialistNodeGroupParams:
    flock_size: int
    n_symbols: int
    gate_input_context_multiplier: float
    gate_input_context_avg_window_size: int
    seq_count: int
    convert_context_to_one_hot: bool


class SpecialistInputs(GroupInputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create("Data input")
        self.context_input = self.create("Context input")


class SpecialistOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Predicted output")
        self.context_feedback = self.create("Context feedback")


class SpecialistNodeGroup(NodeGroupBase[SpecialistInputs, SpecialistOutputs]):
    def __init__(self,
                 params: SpecialistNodeGroupParams,
                 name: str = "SpecialistNodeGroup", ):
        super().__init__(name, inputs=SpecialistInputs(self), outputs=SpecialistOutputs(self))
        self._params = params

        # Create specialist
        specialist_node = ExpertFlockNode(
            ExpertParams(flock_size=self._params.flock_size,
                         n_cluster_centers=self._params.n_symbols,
                         spatial=SpatialPoolerParams(
                             cluster_boost_threshold=100,
                             enable_learning=False
                         ),
                         temporal=TemporalPoolerParams(
                             n_frequent_seqs=50,
                             seq_length=2,
                             seq_lookahead=1,
                             incoming_context_size=3,
                             forgetting_limit=5000,
                             compute_best_matching_context=True
                         ),
                         ),
            name="Specialist"
        )
        self.add_node(specialist_node)
        self._specialist_node = specialist_node

        Connector.connect(self.inputs.input.output, specialist_node.inputs.sp.data_input)
        Connector.connect(self.inputs.context_input.output, specialist_node.inputs.tp.context_input)

        # Context feedback Output

        context_flatten_node = FlattenNode(1)
        context_enhance_node = self.create_node_context_enhance()
        self.add_node(context_flatten_node)
        self.add_node(context_enhance_node)

        Connector.connect(specialist_node.outputs.tp.best_matching_context, context_flatten_node.inputs.input)

        if self._params.convert_context_to_one_hot:
            context_to_one_hot_node = ToOneHotNode(ToOneHotMode.MAX)
            self.add_node(context_to_one_hot_node)
            Connector.connect(context_flatten_node.outputs.output, context_to_one_hot_node.inputs.input)
            Connector.connect(context_to_one_hot_node.outputs.output, context_enhance_node.inputs[0])
        else:
            Connector.connect(context_flatten_node.outputs.output, context_enhance_node.inputs[0])

        Connector.connect(context_enhance_node.outputs[0], self.outputs.context_feedback.input)

        # predicted symbol evaluation
        extract_predicted_output_node = self.create_node_extract_predicted_output()
        self.add_node(extract_predicted_output_node)
        Connector.connect(specialist_node.memory_blocks.tp.passive_predicted_clusters,
                          extract_predicted_output_node.inputs[0])
        predicted_output_to_one_hot_node = ToOneHotNode(ToOneHotMode.MAX, name="Specialist predicted one hot output")
        self.add_node(predicted_output_to_one_hot_node)
        Connector.connect(extract_predicted_output_node.outputs[0], predicted_output_to_one_hot_node.inputs.input)
        Connector.connect(predicted_output_to_one_hot_node.outputs.output, self.outputs.output.input)

    def create_node_extract_predicted_output(self) -> NodeBase:
        specialist_extract_predicted_output_last_value = torch.zeros((self._params.flock_size, self._params.n_symbols))

        def specialist_extract_predicted_output(last_value, inputs, outputs):
            # global context_enhance_avg
            output = inputs[0]

            next_seq_position = self._specialist_node.params.temporal.seq_lookbehind
            clusters = output[:, next_seq_position, :]
            result = clusters.view(self._params.flock_size, self._params.n_symbols)

            outputs[0].copy_(last_value)
            last_value.copy_(result)

        return LambdaNode(partial(specialist_extract_predicted_output, specialist_extract_predicted_output_last_value),
                          1, [(self._params.flock_size, self._params.n_symbols)],
                          name="Specialist extract predicted output")

    def create_node_context_enhance(self) -> NodeBase:
        data_window_list = []

        def context_enhance(data_window: List[torch.Tensor], inputs, outputs):
            """Multiply context by coef and average over n steps"""
            context = inputs[0] * self._params.gate_input_context_multiplier

            # Add value to sliding window
            data_window.append(context)
            if len(data_window) > self._params.gate_input_context_avg_window_size - 1:
                data_window[:] = data_window[-self._params.gate_input_context_avg_window_size:]

            # average values
            outputs[0].copy_(torch.mean(torch.stack(data_window), 0))

        return LambdaNode(partial(context_enhance, data_window_list), 1,
                          [(self._params.flock_size, self._params.seq_count)], name="Gateway input context enhance")

    def init_sp_clusters(self):
        # set specialist SP cluster centers (just identity)
        # n, m = self._specialist_node.memory_blocks.sp.cluster_centers.tensor.shape[-2:]
        # eye = torch.eye(n, m)
        eye = torch.eye(self._params.n_symbols)
        self._specialist_node.memory_blocks.sp.cluster_centers.tensor.copy_(eye)
