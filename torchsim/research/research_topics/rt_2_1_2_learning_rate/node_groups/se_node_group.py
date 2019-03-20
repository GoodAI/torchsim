from typing import List

import torch

from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.eval.node_accessors.random_number_accessor import RandomNumberNodeAccessor
from torchsim.core.eval.node_accessors.se_io_accessor import SeIoAccessor
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.core.nodes import RandomNumberNode, ConstantNode, RandomNoiseNode, RandomNoiseParams
from torchsim.core.nodes.dataset_se_objects_node import DatasetConfig, DatasetSeObjectsParams
from torchsim.research.se_tasks.topologies.se_io.se_io_general import SeIoGeneral
from torchsim.research.se_tasks.topologies.se_io.se_io_task0_dataset import SeIoTask0Dataset
from torchsim.research.se_tasks.topologies.task0_base_topology import Task0BaseGroupOutputs


class SeNodeGroup(NodeGroupBase[EmptyInputs, Task0BaseGroupOutputs]):
    """
    Contains the Task0 dataset and the following baselines:
        - zeros                         [num_labels]
        - random one-hot                [num_labels]
        - random one-hot for each layer [layer_size(layer)] (layer_size ~ sp_output_size)

    The group has two outputs: image and label outputs
    """

    @property
    def is_se_testing_phase(self):
        return self._se_io.get_testing_phase()

    def __init__(self,
                 baseline_seed: int = None,
                 layer_sizes: List[int] = (100, 100),
                 class_filter: List[int] = None,
                 image_size=SeDatasetSize.SIZE_24,
                 random_order: bool = False,
                 noise_amp: float = 0.0,
                 use_se: bool = False  # True: use a running instance of SE for getting data; False: use a dataset
                 ):
        """

        Args:
            baseline_seed:
            layer_sizes:
            class_filter:
            image_size:
            random_order:
            noise_amp:  in case the noise_amp is > 0, superpose noise with mean 0 and variance=noise_amp to the image
        """
        super().__init__("Task 0 - SeNodeGroup", outputs=Task0BaseGroupOutputs(self))

        self.use_se = use_se
        if use_se:
            self._se_io = SeIoGeneral()
            self._se_io.se_config.render_width = image_size.value
            self._se_io.se_config.render_height = image_size.value
        else:
            # dataset and params
            params = DatasetSeObjectsParams()
            params.dataset_config = DatasetConfig.TRAIN_ONLY
            params.dataset_size = image_size
            params.class_filter = class_filter
            params.random_order = random_order
            params.seed = baseline_seed
            self._se_io = SeIoTask0Dataset(params)

        self._se_io.install_nodes(self)

        if use_se:
            blank_task_control = ConstantNode((self._se_io.se_config.TASK_CONTROL_SIZE,))

            actions_node = ConstantNode((4,), name="actions")

            blank_task_labels = ConstantNode((20,), name="labels")

            self.add_node(blank_task_control)
            self.add_node(actions_node)
            self.add_node(blank_task_labels)

            Connector.connect(blank_task_control.outputs.output, self._se_io.inputs.task_control)
            Connector.connect(actions_node.outputs.output, self._se_io.inputs.agent_action)
            Connector.connect(blank_task_labels.outputs.output, self._se_io.inputs.agent_to_task_label)

        # baselines for each layer
        self._baselines = []
        for layer_size in layer_sizes:

            node = RandomNumberNode(upper_bound=layer_size, seed=baseline_seed)
            self.add_node(node)
            self._baselines.append(node)

        # baseline for the labels separately
        self._label_baseline = ConstantNode(shape=self._se_io.get_num_labels(), constant=0, name='label_const')
        self._random_label_baseline = RandomNumberNode(upper_bound=self._se_io.get_num_labels(), seed=baseline_seed)

        self.add_node(self._label_baseline)
        self.add_node(self._random_label_baseline)

        if noise_amp > 0.0:
            # add the noise to the output image?
            _random_noise_params = RandomNoiseParams()
            _random_noise_params.distribution = 'Normal'
            _random_noise_params.amplitude = noise_amp
            self._random_noise = RandomNoiseNode(_random_noise_params)
            self.add_node(self._random_noise)
            Connector.connect(self._se_io.outputs.image_output, self._random_noise.inputs.input)
            Connector.connect(self._random_noise.outputs.output, self.outputs.image.input)
        else:
            Connector.connect(self._se_io.outputs.image_output, self.outputs.image.input)

        Connector.connect(self._se_io.outputs.task_to_agent_label, self.outputs.labels.input)

    # provides access from the template
    def get_label_id(self) -> int:
        """Label - scalar value"""
        return SeIoAccessor.get_label_id(self._se_io)

    def clone_ground_truth_label_tensor(self) -> torch.Tensor:
        """Label - one-hot tensor"""
        return SeIoAccessor.task_to_agent_label_ground_truth(self._se_io).clone()

    def clone_constant_baseline_output_tensor_for_labels(self) -> torch.Tensor:
        """Zeros - constant tensor"""
        return self._label_baseline.outputs.output.tensor.clone()

    def clone_random_baseline_output_tensor_for_labels(self) -> torch.Tensor:
        """Random one-hot tensor"""
        return RandomNumberNodeAccessor.get_output_tensor(self._random_label_baseline).clone()

    def switch_dataset_training(self, training: bool):
        if not self.use_se:
            io_dataset: SeIoTask0Dataset = self._se_io
            io_dataset.node_se_dataset.switch_training(training_on=training, just_hide_labels=False)

    def get_baseline_output_id_for(self, layer_id: int) -> int:
        return RandomNumberNodeAccessor.get_output_id(self._baselines[layer_id])