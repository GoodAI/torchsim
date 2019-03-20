import logging
from typing import Tuple, Optional

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import GroupOutputs
from torchsim.gui.validators import validate_predicate
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.significant_nodes import SpReconstructionLayer
from torchsim.topologies.toyarch_groups.ncm_group_base import ClassificationTaskInputs, ClassificationTaskOutputs, \
    NCMGroupBase

logger = logging.getLogger(__name__)


class NCMR1Group(NCMGroupBase[ClassificationTaskInputs, ClassificationTaskOutputs]):
    """
    Contains N conv layers in a hierarchy and one fully-connected SP flock on top.
    Designed for a classification task like Task0.

    Create the multilayer topology of configuration: N conv layers -> fully connected layer with labels (N>=1)

        Args:
            conv_layers_params: parameters for each layer (or default ones)
            top_layer_params: parameters for the top layer separately
            model_seed: seed of the topology
            num_labels: number of labels is 20 by default
            image_size: small resolution by default
    """

    def __init__(self,
                 conv_layers_params: MultipleLayersParams,
                 top_layer_params: MultipleLayersParams,
                 model_seed: Optional[int] = 321,
                 num_labels: int = 20,
                 image_size: Tuple[int, int, int] = (24, 24, 3),
                 name: str = "Nc1r1Group"):
        super().__init__(ClassificationTaskInputs(self),
                         ClassificationTaskOutputs(self),
                         conv_layers_params,
                         model_seed,
                         image_size,
                         name)

        self._num_labels = num_labels

        validate_predicate(lambda: self._num_labels is not None,
                           "num_labels cannot be None if top layer is used (top_layer_params is not None).")

        # parse to expert params
        self._top_params = top_layer_params.convert_to_expert_params()[0]

        self.top_layer = SpReconstructionLayer(self.output_projection_sizes,
                                               self._num_labels,
                                               sp_params=self._top_params,
                                               name='TOP',
                                               seed=model_seed)
        self.add_node(self.top_layer)

        # Conv[-1] -> Fully
        Connector.connect(
            self.conv_layers[-1].outputs.data,
            self.top_layer.inputs.data)

        # Label -> Fully
        Connector.connect(
            self.inputs.label.output,
            self.top_layer.inputs.label)

        # Fully -> Reconstructed label
        Connector.connect(
            self.top_layer.outputs.label,
            self.outputs.reconstructed_label.input
        )


class ActionTaskOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.predicted_reconstructed_input = self.create("Reconstructed prediction")

