from typing import Optional, Tuple

from torchsim.core.graph.connection import Connector
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.topologies.toyarch_groups.ncmr1_group import ActionTaskOutputs
from torchsim.topologies.toyarch_groups.ncm_group import NCMGroupInputs, NCMGroupBase


class R1NCMGroup(NCMGroupBase[NCMGroupInputs, ActionTaskOutputs]):
    """N layers of flocks, where the first one takes and input and returns its reconstructed prediction -
    used for example for actions."""

    def __init__(self,
                 conv_layers_params: MultipleLayersParams,
                 model_seed: Optional[int] = 321,
                 image_size: Tuple[int, int, int] = (24, 24, 3)):
        super().__init__(
            inputs=NCMGroupInputs(self),
            outputs=ActionTaskOutputs(self),
            conv_layers_params=conv_layers_params,
            model_seed=model_seed,
            image_size=image_size,
            name="R1NCMGroup")

        Connector.connect(
            self.conv_layers[0].expert_flock_nodes[0].outputs.sp.predicted_reconstructed_input,
            self.outputs.predicted_reconstructed_input.input
        )
