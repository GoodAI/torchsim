from abc import abstractmethod, ABC
from typing import TypeVar, NamedTuple, List, Callable

from torchsim.core.eval2.measurement_manager import RunMeasurementManager
from torchsim.core.eval2.train_test_switchable import TrainTestSwitchable
from torchsim.utils.template_utils.train_test_topology_saver import PersistableSaver


class ExperimentComponent:
    """A base class for an experiment component implementing default behavior for components."""

    def before_topology_step(self):
        """Runs before each step of the simulation."""
        pass

    def after_topology_step(self):
        """Runs after each step of the simulation."""
        pass

    def should_end_run(self) -> bool:
        """Called after after_topology_step(). If this returns True, the experiment is stopped."""
        return False

    def calculate_run_results(self):
        """Calculate results of the single experiment run.

        If some additional measurements are gathered here, they should be added to self.run_measurement_manager via
        the add_custom_data method.
        """
        pass


TComponent = TypeVar('TComponent', bound=ExperimentComponent)


class ExperimentController(ExperimentComponent):
    """A controller for the experiment.

    The controller is created for each experiment run and the components for the experiment are then registered with it.
    The components' methods are called in the order of registration.
    """
    _components: List[ExperimentComponent]

    def __init__(self):
        """Initializes the controller."""
        self._components = []

    def register(self, component: TComponent) -> TComponent:
        """Registers a component with the controller.

        Note that the order of registration is kept later when the individual components' methods are invoked.
        """
        self._components.append(component)
        return component

    def should_end_run(self) -> bool:
        for component in self._components:
            if component.should_end_run():
                return True
        return False

    def before_topology_step(self):
        for component in self._components:
            component.before_topology_step()

    def after_topology_step(self):
        for component in self._components:
            component.after_topology_step()

    def calculate_run_results(self):
        for component in self._components:
            component.calculate_run_results()


class TrainTestComponentParams(NamedTuple):
    """Parameters fo the Train/Test component."""
    num_testing_phases: int
    num_testing_steps: int
    overall_training_steps: int

    @property
    def max_steps(self):
        return self.num_testing_phases * self.num_testing_steps + self.overall_training_steps


class TrainTestMeasuringComponent(ExperimentComponent, ABC):
    def __init__(self, run_measurement_manager: RunMeasurementManager):
        self._run_measurement_manager = run_measurement_manager

    def add_measurement_f_training(self, item_name, m_function: Callable, period: int = 1):
        """Runs the measurement with given period only during training."""
        self._run_measurement_manager.add_measurement_f(item_name,
                                                        m_function,
                                                        period=period,
                                                        predicate=self.is_in_training_phase)

    def add_measurement_f_testing(self, item_name, m_function: Callable, period: int = 1):
        """Runs the measurement with given period only during testing."""
        self._run_measurement_manager.add_measurement_f(item_name,
                                                        m_function,
                                                        period=period,
                                                        predicate=self.is_in_testing_phase)

    @abstractmethod
    def is_in_training_phase(self) -> bool:
        pass

    @abstractmethod
    def is_in_testing_phase(self) -> bool:
        pass


class TrainTestControllingComponent(TrainTestMeasuringComponent):
    """A component which controls and keeps track of training and testing phases.

    If you have a reference to this component, you can use the add_measurement_f_training and _testing methods
    to add measurements only for the respective phases.
    """
    _training_step: int  # ID of the currently performed step (the one that will be done after _do_before_topology_step)
    _testing_step: int
    _training_phase_id: int  # ID of the currently performed phase
    _testing_phase_id: int

    _is_in_training_phase: bool  # whether we are in the training phase now

    _topology_saver: PersistableSaver

    def __init__(self, switchable: TrainTestSwitchable, run_measurement_manager: RunMeasurementManager,
                 train_test_params: TrainTestComponentParams):
        super().__init__(run_measurement_manager)
        self._switchable = switchable
        self._training_step = 0
        self._testing_step = 0
        self._training_phase_id = 0
        self._testing_phase_id = 0

        self._is_in_training_phase = True

        self._num_testing_phases = train_test_params.num_testing_phases
        self._num_testing_steps = train_test_params.num_testing_steps

        self._training_steps_between_testing = train_test_params.overall_training_steps // self._num_testing_phases
        self._topology_saver = PersistableSaver(type(self._switchable).__name__)

        run_measurement_manager.add_measurement_f('testing_phase_id', self.testing_phase_id_f, period=1)
        run_measurement_manager.add_measurement_f('training_phase_id', self.training_phase_id_f, period=1)
        run_measurement_manager.add_measurement_f('testing_step', self.testing_step_f, period=1)
        run_measurement_manager.add_measurement_f('training_step', self.training_step_f, period=1)

        self._first_step = True
        self._setup_initial_training()

    def _setup_initial_training(self):
        self._testing_phase_id = -1
        self._testing_step = -1

        self._is_in_training_phase = True
        self._training_phase_id = 0
        self._training_step = -1

    def before_topology_step(self):
        super().before_topology_step()

        if self._first_step:
            if self._is_in_training_phase:
                # First step - switch the topology to training.
                # This is here so that we are sure the topology is in training mode at the beginning of the run.
                self._switchable.switch_to_training()
            else:
                self._switchable.switch_to_testing()

            self._first_step = False

        if self._is_in_training_phase:
            # decide whether to do the training for: self._training_step + 1
            if (self._training_step + 1) % self._training_steps_between_testing == 0 and self._training_step != -1:
                self._topology_saver.save_data_of(self._switchable)

                self._switchable.switch_to_testing()
                self._is_in_training_phase = False
                self._testing_phase_id += 1  # ID of the current phase being measured
        else:
            if (self._testing_step + 1) % self._num_testing_steps == 0 and self._testing_step != -1:
                self._topology_saver.load_data_into(self._switchable)

                self._switchable.switch_to_training()
                self._is_in_training_phase = True
                self._training_phase_id += 1

        if self._is_in_training_phase:
            self._training_step += 1
        else:
            self._testing_step += 1

    def testing_phase_id_f(self):
        if self._is_in_training_phase:
            return -1
        return self._testing_phase_id

    def training_phase_id_f(self):
        if not self._is_in_training_phase:
            return -1
        return self._training_phase_id

    def testing_step_f(self):
        if self._is_in_training_phase:
            return -1
        return self._testing_step

    def training_step_f(self):
        if not self._is_in_training_phase:
            return -1
        return self._training_step

    def is_in_training_phase(self):
        return self._is_in_training_phase

    def is_in_testing_phase(self):
        return not self.is_in_training_phase()
