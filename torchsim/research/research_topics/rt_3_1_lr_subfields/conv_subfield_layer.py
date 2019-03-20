from typing import Tuple

from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import ConvSpatialPoolerFlockNode
from torchsim.research.research_topics.rt_3_1_lr_subfields.random_subfield_node import RandomSubfieldForkNode
from torchsim.significant_nodes import ConvLayer


class SubLrfConvLayer(ConvLayer):
    SHORT_NAME = "SFC"

    def __init__(self,
                 input_dims,
                 rf_output_dims,
                 stride: Tuple[int, int] = None,
                 expert_params: ExpertParams = None,
                 num_flocks: int = 1,
                 name="",
                 seed: int = None,
                 sub_field_size=6):
        super().__init__(input_dims,
                         rf_output_dims,
                         stride,
                         expert_params,
                         num_flocks,
                         name,
                         seed)

        n_sub_fields = len(self.expert_flock_nodes)

        subfield_node = RandomSubfieldForkNode(n_outputs=n_sub_fields, n_samples=sub_field_size,
                                               first_non_expanded_dim=-3)
        self.add_node(subfield_node)
        Connector.connect(self.lrf_node.outputs.output, subfield_node.inputs.input)

        for i, expert_flock in enumerate(self.expert_flock_nodes):
            Connector.disconnect_input(expert_flock.inputs.sp.data_input)
            Connector.connect(subfield_node.outputs[i], expert_flock.inputs.sp.data_input)


class SubLrfSpConvLayer(SubLrfConvLayer):
    conv_expert_node_class = ConvSpatialPoolerFlockNode
    SHORT_NAME = "SFSC"

