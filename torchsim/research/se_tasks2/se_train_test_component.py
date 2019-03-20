from abc import ABC, abstractmethod

from torchsim.core.eval2.experiment_controller import TrainTestMeasuringComponent
from torchsim.core.eval2.measurement_manager import RunMeasurementManager
from torchsim.core.eval2.train_test_switchable import TrainTestSwitchable
from torchsim.research.research_topics.rt_2_1_2_learning_rate.node_groups.se_node_group import SeNodeGroup


class SENodeGroupProvider(ABC):
    @property
    @abstractmethod
    def se_node_group(self) -> SeNodeGroup:
        pass


class SETrainTestModel(TrainTestSwitchable, SENodeGroupProvider, ABC):
    pass


class SETrainTestComponent(TrainTestMeasuringComponent):
    """A component that tracks the train/test phases for Space Engineers."""

    def __init__(self, se_train_test_model: SETrainTestModel, run_measurement_manager: RunMeasurementManager):
        super().__init__(run_measurement_manager)
        self._se_train_test_model = se_train_test_model
        self._is_in_testing_phase = False

    def after_topology_step(self):
        """Switches the training/testing phase in the topology.

        Note that this is done with a 1-step delay if the backwards (action) edge is between
        the agent and the environment.
        """
        super().after_topology_step()
        is_current_testing_phase = self._se_train_test_model.se_node_group.is_se_testing_phase
        # Detect change.
        if self._is_in_testing_phase != is_current_testing_phase:
            if is_current_testing_phase:
                self._se_train_test_model.switch_to_testing()
            else:
                self._se_train_test_model.switch_to_training()

        self._is_in_testing_phase = is_current_testing_phase

    def is_in_training_phase(self) -> bool:
        return not self.is_in_testing_phase()

    def is_in_testing_phase(self) -> bool:
        return self._is_in_testing_phase
