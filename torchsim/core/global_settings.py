from torchsim.core.exceptions import PrivateConstructorException, IllegalStateException
from typing import TYPE_CHECKING, Optional, Callable

if TYPE_CHECKING:
    from torchsim.core.experiment_runner import UiExperimentRunner


class GlobalSettings:
    _instance: 'GlobalSettings' = None

    observer_memory_block_minimal_size: int

    def __init__(self):
        self.observer_memory_block_minimal_size = 50
        # Emulate private constructor
        raise PrivateConstructorException(self)

    @classmethod
    def instance(cls) -> 'GlobalSettings':
        if cls._instance is None:
            try:
                _ = GlobalSettings()
            except PrivateConstructorException as e:
                # Emulate private constructor
                cls._instance = e.instance

        return cls._instance


class SimulationThreadRunner:
    _instance: 'SimulationThreadRunner' = None
    _simulation: Optional['UiExperimentRunner']

    def __init__(self):

        # Emulate private constructor
        raise PrivateConstructorException(self)

    @classmethod
    def instance(cls) -> 'SimulationThreadRunner':
        if cls._instance is None:
            try:
                _ = SimulationThreadRunner()
            except PrivateConstructorException as e:
                # Emulate private constructor
                cls._instance = e.instance

        return cls._instance

    def _get_simulation(self) -> 'UiExperimentRunner':
        if self._simulation is None:
            raise IllegalStateException(f'Simulation is not set')
        return self._simulation

    def set_simulation(self, value: 'UiExperimentRunner'):
        self._simulation = value

    def run_in_simulation_thread(self, runnable: Callable):
        if self._get_simulation().is_running():
            self._get_simulation().add_to_run_queue(runnable)
        else:
            # When simulation is not running, execute the code immediately in caller thread.
            # It is safe now.
            runnable()
