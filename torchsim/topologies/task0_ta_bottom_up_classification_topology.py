from typing import Optional, List

from eval_utils import run_just_model
from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import DatasetSeObjectsNode, DatasetSeObjectsParams, DatasetConfig
from torchsim.core.nodes.bottom_up_attention_group import BottomUpAttentionGroup
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group_params import \
    MultipleLayersParams
from torchsim.significant_nodes import SpReconstructionLayer


class Task0TaBottomUpClassificationTopology(Topology):
    """Bottom-up focusing mechanism produces image patches, these are sent to the SPReconstruction with labels"""

    _node_se_dataset: DatasetSeObjectsNode
    _attention_group: BottomUpAttentionGroup
    _sp_reconstruction: SpReconstructionLayer

    _dataset_params: DatasetSeObjectsParams
    _reconstruction_params: ExpertParams

    _label_size: int
    _input_data_size: int

    _fof_fixed_size: int

    def __init__(self,
                 top_layer_params: MultipleLayersParams = None,
                 model_seed: Optional[int] = None,
                 baseline_seed: Optional[int] = None,
                 class_filter: List[int] = None,
                 random_order: bool = False,
                 num_labels: int = 20,
                 image_size=SeDatasetSize.SIZE_64,
                 fof_fixed_size: Optional[int] = None):
        super().__init__(device='cuda')

        # parse params here
        self._dataset_params = DatasetSeObjectsParams()
        self._dataset_params.dataset_config = DatasetConfig.TRAIN_ONLY
        self._dataset_params.dataset_size = image_size
        self._dataset_params.class_filter = class_filter
        self._dataset_params.random_order = random_order
        self._dataset_params.seed = baseline_seed

        if top_layer_params is None:
            top_layer_params = MultipleLayersParams()
        self._top_params = top_layer_params.convert_to_expert_params()[0]

        self._label_size = num_labels
        self._fof_fixed_size = fof_fixed_size

        # create and add nodes here
        self._node_se_dataset = DatasetSeObjectsNode(self._dataset_params)
        self.add_node(self._node_se_dataset)

        self._attention_group = BottomUpAttentionGroup()
        self.add_node(self._attention_group)

        if self._fof_fixed_size is not None:
            input_size = self._fof_fixed_size
            self._attention_group.fixed_region_size = self._fof_fixed_size
            self._attention_group.use_fixed_region = True
        else:
            input_size = image_size.value
            self._attention_group.use_fixed_region = False
        self._input_data_size = input_size * input_size * 3

        self._sp_reconstruction_layer = SpReconstructionLayer(
            self._input_data_size,
            self._label_size,
            self._top_params,
            "ReconstructionLayer",
            model_seed)
        self.add_node(self._sp_reconstruction_layer)

        # connect nodes here
        Connector.connect(self._node_se_dataset.outputs.image_output,
                          self._attention_group.inputs.image)
        Connector.connect(self._attention_group.outputs.fof,
                          self._sp_reconstruction_layer.inputs.data)
        Connector.connect(self._node_se_dataset.outputs.task_to_agent_label,
                          self._sp_reconstruction_layer.inputs.label)

    def switch_learning(self, on: bool):
        self._node_se_dataset.switch_training(training_on=on, just_hide_labels=False)
        self._sp_reconstruction_layer.switch_learning(on)


if __name__ == '__main__':
    """Just an example configuration for GUI"""

    expert_params = MultipleLayersParams()
    expert_params.n_cluster_centers = 200
    expert_params.sp_buffer_size = 3000
    expert_params.sp_batch_size = 1000
    expert_params.learning_rate = 0.05
    expert_params.cluster_boost_threshold = 1000
    expert_params.compute_reconstruction = True

    class_f = None

    params = [
        {
            'top_layer_params': expert_params,
            'image_size': SeDatasetSize.SIZE_64,
            'class_filter': class_f,
            'model_seed': None,
            'baseline_seed': None,
            'random_order': False,
            'fof_fixed_size': None
        }
    ]

    run_just_model(Task0TaBottomUpClassificationTopology(**params[0]), gui=True, persisting_observer_system=True)
