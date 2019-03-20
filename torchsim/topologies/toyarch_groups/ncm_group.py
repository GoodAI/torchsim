import logging
from typing import Tuple, Optional

from torchsim.core.graph.node_base import EmptyOutputs
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.topologies.toyarch_groups.ncm_group_base import NCMGroupInputs, NCMGroupBase

logger = logging.getLogger(__name__)


class NCMGroup(NCMGroupBase[NCMGroupInputs, EmptyOutputs]):
    def __init__(self,
                 conv_layers_params: MultipleLayersParams,
                 model_seed: Optional[int] = 321,
                 image_size: Tuple[int, int, int] = (24, 24, 3),
                 name: str = "TA Model"):
        super().__init__(NCMGroupInputs(self),
                         EmptyOutputs(self),
                         conv_layers_params,
                         model_seed,
                         image_size,
                         name)
