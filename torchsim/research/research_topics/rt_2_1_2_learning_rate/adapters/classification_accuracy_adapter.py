from abc import ABC, abstractmethod

import torch

from torchsim.core.eval.node_accessors.random_number_accessor import RandomNumberNodeAccessor
from torchsim.core.eval.node_accessors.se_io_accessor import SeIoAccessor
from torchsim.core.graph import Topology
from torchsim.core.nodes.random_number_node import RandomNumberNode
from torchsim.core.nodes.constant_node import ConstantNode
from torchsim.research.experiment_templates.task0_train_test_classification_acc_template import \
    Task0TrainTestClassificationAccAdapter
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.topologies.task0_base_topology import Task0BaseTopology
from torchsim.research.se_tasks.topologies.se_io.se_io_base import SeIoBase
from torchsim.research.se_tasks.topologies.se_io.se_io_task0_dataset import SeIoTask0Dataset


class ClassificationAccuracyAdapterBase(Task0TrainTestClassificationAccAdapter, ABC):
    """Compute classification accuracy of the topology on the test set of the task 0 dataset"""

    _topology: Task0BaseTopology

    _se_io: SeIoBase
    _label_baseline: ConstantNode
    _random_label_baseline: RandomNumberNode

    _is_training: bool

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

    def is_in_training_phase(self, **kwargs) -> bool:
        # print(f'is training: {self._topology._is_learning} sim step: {self._topology._current_step}')
        # return self._topology._is_learning
        return self._is_training

    def _switch_dataset_training(self, training: bool):
        # SE probably do not support manual switching between train/test
        assert type(self._se_io) is SeIoTask0Dataset
        io_dataset: SeIoTask0Dataset = self._se_io
        io_dataset.node_se_dataset.switch_training(training_on=training, just_hide_labels=False)

    @abstractmethod
    def model_switch_to_training(self):
        """Switch experts (NN) to training"""
        pass

    @abstractmethod
    def model_switch_to_testing(self):
        """Switch experts (NN) to testing"""
        pass

    def switch_to_training(self):
        self._is_training = True
        self._switch_dataset_training(True)
        self.model_switch_to_training()

    def switch_to_testing(self):
        self._is_training = False
        self._switch_dataset_training(False)
        self.model_switch_to_testing()

    def get_topology(self) -> Topology:
        return self._topology

    def set_topology(self, topology: Task0BaseTopology):
        """Sets the topology, override to add any custom functionality"""

        self._topology = topology

        self._se_io = self._topology.se_io
        self._label_baseline = self._topology.label_baseline
        self._random_label_baseline = self._topology.random_label_baseline


