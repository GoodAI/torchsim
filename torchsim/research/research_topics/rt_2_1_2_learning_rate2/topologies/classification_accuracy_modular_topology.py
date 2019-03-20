import torch
from eval_utils import run_just_model
from torchsim.core.eval2.scaffolding import TopologyScaffoldingFactory
from torchsim.core.eval2.train_test_switchable import TrainTestSwitchable
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector

from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.se_node_group import SeNodeGroup
from torchsim.research.research_topics.rt_2_1_2_learning_rate2.node_groups.classification_model_group import \
    ClassificationModelGroup
from torchsim.research.research_topics.rt_2_1_2_learning_rate2.node_groups.ta_multilayer_node_group import \
    Nc1r1ClassificationGroup


class ClassificationAccuracyModularTopology(Topology, TrainTestSwitchable):
    """New version of the modular topology for a classification task."""

    _is_training: bool

    @property
    def se_node_group(self):
        return self._se_group

    def __init__(self, se_group: SeNodeGroup, model: ClassificationModelGroup):
        super().__init__('cuda')
        self._se_group = se_group
        self._model = model

        self.add_node(self._se_group)
        self.add_node(self._model)

        Connector.connect(
            self._se_group.outputs.image,
            self._model.inputs.image
        )

        Connector.connect(
            self._se_group.outputs.labels,
            self._model.inputs.label
        )

    def get_label_id(self) -> int:
        return self._se_group.get_label_id()

    def clone_ground_truth_label_tensor(self) -> torch.Tensor:
        return self._se_group.clone_ground_truth_label_tensor()

    def clone_constant_baseline_output_tensor_for_labels(self) -> torch.Tensor:
        return self._se_group.clone_constant_baseline_output_tensor_for_labels()

    def clone_random_baseline_output_tensor_for_labels(self) -> torch.Tensor:
        return self._se_group.clone_random_baseline_output_tensor_for_labels()

    def get_average_log_delta_for(self, layer_id: int) -> float:
        return self._model.get_average_log_delta_for(layer_id)

    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        return self._model.clone_predicted_label_tensor_output()

    # ################ common methods
    def switch_to_training(self):
        self._is_training = True
        self._se_group.switch_dataset_training(True)
        self._model.model_switch_to_training()

    def switch_to_testing(self):
        self._is_training = False
        self._se_group.switch_dataset_training(False)
        self._model.model_switch_to_testing()


if __name__ == '__main__':
    params = [
        {'se_group': {'class_filter': (1, 2, 3, 4)}, 'model': {'num_cc': (100, 230), 'experts_on_x': (2,)}}
    ]

    scaffolding = TopologyScaffoldingFactory(ClassificationAccuracyModularTopology, se_group=SeNodeGroup,
                                             model=Nc1r1ClassificationGroup)

    run_just_model(scaffolding.create_topology(**params[0]), gui=True, persisting_observer_system=True)
