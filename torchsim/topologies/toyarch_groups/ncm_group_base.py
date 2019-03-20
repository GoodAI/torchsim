import logging
from abc import ABCMeta
from typing import TypeVar, Tuple, Optional, List

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_base import TOutputs
from torchsim.core.graph.node_group import GroupInputs, NodeGroupBase, GroupOutputs
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import LambdaNode
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.significant_nodes.conv_layer import create_connected_conv_layers, ConvLayerParams

logger = logging.getLogger(__name__)


class NCMGroupInputs(GroupInputs):

    def __init__(self, owner):
        super().__init__(owner)
        self.image = self.create("Data")


TNCMGroupInputs = TypeVar('TNCMGroupInputs', bound=NCMGroupInputs)


class NCMGroupBase(NodeGroupBase[TNCMGroupInputs, TOutputs]):
    """ Contains N conv layers in a hierarchy.
    """

    _rescale_node: LambdaNode
    _num_labels: int
    _num_layers: int
    _input_dims: Tuple[int, int, int]

    def __init__(self,
                 inputs: TNCMGroupInputs,
                 outputs: TOutputs,
                 conv_layers_params: MultipleLayersParams,
                 model_seed: Optional[int] = 321,
                 image_size: Tuple[int, int, int] = (24, 24, 3),
                 name: str = "TA Model"):
        """ Create the multilayer topology of configuration: N conv layers

        Args:
            conv_layers_params: parameters for each layer (or default ones)
            model_seed: seed of the topology
            image_size: small resolution by default
        """
        super().__init__(name=name,
                         inputs=inputs,
                         outputs=outputs)

        self._num_layers = conv_layers_params.num_conv_layers + 1
        self._input_dims = image_size

        # parse conv layers to ExpertParams
        self._conv_params_list = conv_layers_params.convert_to_expert_params()

        # other parameters for the convolution
        conv_classes = conv_layers_params.read_list_of_params('conv_classes')
        rf_sizes = conv_layers_params.read_list_of_params('rf_size')
        rf_strides = conv_layers_params.read_list_of_params('rf_stride')
        num_flocks = conv_layers_params.read_list_of_params('n_flocks')

        self._conv_params = self.create_conv_layer_params(param_list=self._conv_params_list,
                                                          rf_sizes=rf_sizes,
                                                          rf_strides=rf_strides,
                                                          num_flocks=num_flocks,
                                                          conv_classes=conv_classes,
                                                          model_seed=model_seed)

        self.conv_layers, self.output_projection_sizes = create_connected_conv_layers(self._conv_params,
                                                                                      self._input_dims)
        for layer in self.conv_layers:
            self.add_node(layer)

        # image -> Conv0
        Connector.connect(
            self.inputs.image.output,
            self.conv_layers[0].inputs.data)

    @staticmethod
    def create_conv_layer_params(
            param_list: List[ExpertParams],
            rf_sizes: List[Tuple[int, int]],
            rf_strides: List[Tuple[int, int]],
            num_flocks: List[int],
            conv_classes: List[ABCMeta],
            model_seed: int = None):
        conv_params = []

        for layer_id, param in enumerate(param_list):
            par = ConvLayerParams(expert_params=param,
                                  input_rf_sizes=rf_sizes[layer_id],
                                  input_rf_stride=rf_strides[layer_id],
                                  num_flocks=num_flocks[layer_id],
                                  name="L" + str(layer_id),
                                  conv_layer_class=conv_classes[layer_id],
                                  seed=model_seed
                                  )
            conv_params.append(par)
        return conv_params


class ClassificationTaskInputs(NCMGroupInputs):

    def __init__(self, owner):
        super().__init__(owner)
        self.label = self.create("Label")


class ClassificationTaskOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.reconstructed_label = self.create("Reconstructed Label")
