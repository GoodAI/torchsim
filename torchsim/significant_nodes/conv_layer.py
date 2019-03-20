from abc import ABC, abstractmethod

from typing import Tuple, List, Dict, Any, Callable

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import NodeGroupBase, GroupInputs, GroupOutputs
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import ConvExpertFlockNode, ReceptiveFieldNode, ConvSpatialPoolerFlockNode, JoinNode
from torchsim.utils.list_utils import dim_prod


class LayerInputs(GroupInputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.data = self.create("Data")


class LayerOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.data = self.create("Data")


class TaLayer(NodeGroupBase[LayerInputs, LayerOutputs], ABC):

    @abstractmethod
    def switch_learning(self, on: bool):
        pass


class ConvLayer(TaLayer):
    conv_expert_node_class = ConvExpertFlockNode
    expert_flock_nodes: List[ConvExpertFlockNode]
    SHORT_NAME = "C"

    def __init__(self,
                 input_dims,
                 rf_output_dims,
                 stride: Tuple[int, int] = None,
                 expert_params: ExpertParams = None,
                 num_flocks: int = 1,
                 name="",
                 seed: int = None):
        super().__init__(name + " ConvLayer", inputs=LayerInputs(self), outputs=LayerOutputs(self))

        lrf_node = ReceptiveFieldNode(input_dims=input_dims, parent_rf_dims=rf_output_dims,
                                      parent_rf_stride=stride, name=name + " LRF")
        self.add_node(lrf_node)
        self.lrf_node = lrf_node
        self.input_data = lrf_node.inputs.input
        Connector.connect(self.inputs.data.output, lrf_node.inputs.input)

        if expert_params is None:
            expert_params = ExpertParams()

        self.expert_flock_nodes = []

        if num_flocks == 1:
            # create one flock only
            expert_flock_node = self.conv_expert_node_class(expert_params, name=name + " Expert", seed=seed)
            self.add_node(expert_flock_node)
            self.expert_flock_nodes.append(expert_flock_node)
            Connector.connect(lrf_node.outputs.output, expert_flock_node.inputs.sp.data_input)
            Connector.connect(self._get_output_memory_block(expert_flock_node), self.outputs.data.input)
        else:
            # create more flocks and join node
            join_node = JoinNode(dim=len(input_dims) - 1, n_inputs=num_flocks, name=name + " Join")
            self.add_node(join_node)
            self.join_node = join_node

            for i in range(num_flocks):
                expert_flock_node = self.conv_expert_node_class(expert_params, name=name + f" Expert {i}", seed=seed)
                self.add_node(expert_flock_node)
                self.expert_flock_nodes.append(expert_flock_node)
                Connector.connect(lrf_node.outputs.output, expert_flock_node.inputs.sp.data_input)

                Connector.connect(self._get_output_memory_block(expert_flock_node), join_node.inputs[i])

            Connector.connect(join_node.outputs.output, self.outputs.data.input)

    def switch_learning(self, on: bool):
        for expert_flock_node in self.expert_flock_nodes:
            expert_flock_node.switch_learning(on)

    @staticmethod
    def _get_output_memory_block(expert_flock_node: ConvExpertFlockNode):
        return expert_flock_node.outputs.tp.projection_outputs


class SpConvLayer(ConvLayer):
    conv_expert_node_class = ConvSpatialPoolerFlockNode
    expert_flock_node: ConvSpatialPoolerFlockNode
    SHORT_NAME = "SC"

    @staticmethod
    def _get_output_memory_block(expert_flock_node: ConvSpatialPoolerFlockNode):
        return expert_flock_node.outputs.sp.forward_clusters


class ConvLayerParams:
    expert_params: ExpertParams
    input_rf_sizes: Tuple[int, int]  # Y,X sizes of the receptive field of the expert
    input_rf_stride: Tuple[int, int]
    num_flocks: int
    name: str
    use_sp_only: bool
    seed: int
    custom_args: Dict[str, Any]

    def __init__(self,
                 expert_params: ExpertParams,
                 input_rf_sizes: Tuple[int, int],
                 name: str,
                 num_flocks: int = 1,
                 input_rf_stride: Tuple[int, int] = None,
                 conv_layer_class: Callable[..., ConvLayer] = ConvLayer,
                 seed: int = None,
                 custom_args: Dict[str, Any] = None):
        """

        Args:
            expert_params:
            input_rf_sizes: size of the receptive field of each expert
            name:
            num_flocks: redundant representations, how many flocks to create?
            input_rf_stride: stride with which to move the receptive field of experts
            conv_layer_class: which conv layer to use? (SpConvLayer or ConvLayer)
            seed:
            custom_args:
        """
        self.expert_params = expert_params
        self.input_rf_sizes = input_rf_sizes
        self.input_rf_stride = input_rf_stride
        self.num_flocks = num_flocks
        self.name = name
        self.conv_layer_class = conv_layer_class
        self.seed = seed
        self.custom_args = dict() if custom_args is None else custom_args


def compute_grid_size(input_dims: Tuple[int, int, int],
                      input_rf_sizes: Tuple[int, int],
                      input_rf_stride: Tuple[int, int] = None) -> Tuple[int, int]:
    """ Compute grid size given the input_dims (3D), rf_sizes (2D) and optional rf_stride (2D)
    Args:
        input_dims: dimensions of the input [Y, X, channels (either RGB or SP_Size)]
        input_rf_sizes: size of receptive field of each expert in this layer
        input_rf_stride: stride with which the receptive fields shift

    Returns: returns counts of experts on each axis [Y, X]. The final output size is Y*X*SP_Size
    """

    if input_rf_stride is None:
        assert input_dims[0] % input_rf_sizes[0] == 0
        assert input_dims[1] % input_rf_sizes[1] == 0
        grid_size = (input_dims[0] // input_rf_sizes[0], input_dims[1] // input_rf_sizes[1])

    else:
        assert (input_dims[0] - input_rf_sizes[0]) % input_rf_stride[0] == 0
        assert (input_dims[1] - input_rf_sizes[1]) % input_rf_stride[1] == 0
        grid_size = ((input_dims[0] - input_rf_sizes[0]) // input_rf_stride[0] + 1,
                     (input_dims[1] - input_rf_sizes[1]) // input_rf_stride[1] + 1)

    return grid_size


def create_conv_layer(layer_params: ConvLayerParams, input_dims: Tuple[int, int, int]) -> Tuple[ConvLayer, Tuple[int, ...]]:
    """ Sets flock_size in expert_params.

    Args:
        layer_params:
        input_dims:

    Returns:
        conv_layer, expert_params, output_projection_size
    """
    lp = layer_params

    grid_size = compute_grid_size(input_dims, lp.input_rf_sizes, lp.input_rf_stride)

    lp.expert_params.flock_size = grid_size[0] * grid_size[1]

    conv_layer = lp.conv_layer_class(input_dims,
                                     lp.input_rf_sizes,
                                     stride=lp.input_rf_stride,
                                     expert_params=lp.expert_params.clone(),
                                     num_flocks=lp.num_flocks,
                                     name=lp.name,
                                     seed=layer_params.seed,
                                     **lp.custom_args)

    output_size = grid_size + (lp.expert_params.n_cluster_centers * lp.num_flocks,)

    return conv_layer, output_size


def create_connected_conv_layers(layers_params: List[ConvLayerParams], input_dims: Tuple[int, int, int]):
    conv_layers = []
    for i, layer_params in enumerate(layers_params):
        conv_layer, input_dims = \
            create_conv_layer(layer_params, input_dims)
        conv_layers.append(conv_layer)

    for output_layer, input_layer in zip(conv_layers, conv_layers[1:]):
        Connector.connect(output_layer.outputs.data, input_layer.inputs.data)

    return conv_layers, dim_prod(input_dims)
