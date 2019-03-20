import os
from typing import Tuple

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.nodes import RandomNumberNode, ConstantNode, JoinNode
from torchsim.core.nodes.focus_node import FocusNode
from torchsim.core.nodes.grid_world_node import GridWorldNode
from torchsim.core.nodes.images_dataset_node import ImagesDatasetNode, ImagesDatasetParams
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.significant_nodes import SpConvLayer
from torchsim.topologies.toyarch_groups.ncm_group import NCMGroup


class LrfObjectDetectionTopology(Topology):
    """Topology for recognizing individual objects using deep hierarchy with LRFs."""
    _node_grid_world: GridWorldNode

    def __init__(self):
        super().__init__("cuda")

        images_dataset_params = ImagesDatasetParams()
        images_dataset_params.images_path = os.path.join('data', 'datasets', 'image_datasets', 'landmark_world')
        node_images_dataset = ImagesDatasetNode(images_dataset_params)

        # GridWorld sizes
        # egocentric
        width = 160
        height = 95
        fov_size = 16
        fov_half_size = fov_size // 2

        self.add_node(node_images_dataset)

        # to extract an image
        focus_node = FocusNode()
        focus_node._params.trim_output = True
        focus_node._params.trim_output_size = fov_size
        self.add_node(focus_node)

        rnx = RandomNumberNode(upper_bound=width - fov_size)
        self.add_node(rnx)
        rny = RandomNumberNode(upper_bound=height - fov_size)
        self.add_node(rny)
        constant_node = ConstantNode(shape=(2, 1), constant=fov_size)
        self.add_node(constant_node)

        join_node = JoinNode(n_inputs=3, flatten=True)
        self.add_node(join_node)

        # create FOV position and shape
        Connector.connect(rny.outputs.scalar_output, join_node.inputs[0])
        Connector.connect(rnx.outputs.scalar_output, join_node.inputs[1])
        Connector.connect(constant_node.outputs.output, join_node.inputs[2])

        Connector.connect(join_node.outputs.output, focus_node.inputs.coordinates)
        Connector.connect(node_images_dataset.outputs.output_image, focus_node.inputs.input_image)

        self._create_and_connect_agent(focus_node.outputs.focus_output, (fov_size, fov_size, 3))

    def _create_and_connect_agent(self, input_image: MemoryBlock, input_size: Tuple[int, int, int]):

        params = MultipleLayersParams()
        params.num_conv_layers = 4
        params.n_flocks = [5, 5, 1, 1]
        params.n_cluster_centers = [30, 60, 60, 9]
        params.compute_reconstruction = True
        params.conv_classes = SpConvLayer
        params.sp_buffer_size = 5000
        params.sp_batch_size = 500
        params.learning_rate = 0.1
        params.cluster_boost_threshold = 1000
        params.max_encountered_seqs = 1000
        params.max_frequent_seqs = 500
        params.seq_lookahead = 2
        params.seq_length = 4
        params.exploration_probability = 0
        params.rf_size = (2, 2)
        params.rf_stride = None
        ta_group = NCMGroup(conv_layers_params=params, model_seed=None, image_size=input_size)

        self.add_node(ta_group)

        Connector.connect(
            input_image,
            ta_group.inputs.image
        )






