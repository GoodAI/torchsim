import torch

from torchsim.core.graph import Topology
from torchsim.research.experiment_templates.task0_train_test_classification_acc_template import \
    Task0TrainTestClassificationAccAdapter
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.se_node_group import SeNodeGroup
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.ta_multilayer_node_group import \
    Nc1r1GroupWithAdapter
from torchsim.research.research_topics.rt_2_1_2_learning_rate.topologies.task0_ta_se_topology import Task0TaSeTopology


class ClassificationAccuracyModularAdapter(Task0TrainTestClassificationAccAdapter):
    """This is just a temporary solution, which is forward compatible with new  TemplateBase(s)

    Used in the classification template
    """

    _topology: Task0TaSeTopology

    _se_node_group: SeNodeGroup
    _model: Nc1r1GroupWithAdapter

    _is_training: bool

    # ################### just a pass-thru for now, TODO use splitted access in the template
    def get_label_id(self) -> int:
        return self._se_node_group.get_label_id()

    def clone_ground_truth_label_tensor(self) -> torch.Tensor:
        return self._se_node_group.clone_ground_truth_label_tensor()

    def clone_constant_baseline_output_tensor_for_labels(self) -> torch.Tensor:
        return self._se_node_group.clone_constant_baseline_output_tensor_for_labels()

    def clone_random_baseline_output_tensor_for_labels(self) -> torch.Tensor:
        return self._se_node_group.clone_random_baseline_output_tensor_for_labels()

    def is_in_training_phase(self, **kwargs) -> bool:
        # print(f'is training: {self._topology._is_learning} sim step: {self._topology._current_step}')
        # return self._topology._is_learning
        return self._is_training

    # ################# pass thru for the model
    def get_average_log_delta_for(self, layer_id: int) -> float:
        return self._model.get_average_log_delta_for(layer_id)

    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        return self._model.clone_predicted_label_tensor_output()

    # ################ common methods
    def switch_to_training(self):
        self._is_training = True
        self._se_node_group.switch_dataset_training(True)
        self._model.model_switch_to_training()

    def switch_to_testing(self):
        self._is_training = False
        self._se_node_group.switch_dataset_training(False)
        self._model.model_switch_to_testing()

    def get_topology(self) -> Topology:
        return self._topology

    def set_topology(self, topology):
        """Sets the topology, override to add any custom functionality"""

        self._topology = topology
        self._se_node_group = self._topology.se_group
        self._model = self._topology.model

    def is_learning(self) -> bool:
        return self._model.model_is_learning()

