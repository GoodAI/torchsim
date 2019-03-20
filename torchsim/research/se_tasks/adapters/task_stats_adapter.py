from abc import abstractmethod

from torchsim.research.experiment_templates.dataset_simulation_running_stats_template import \
    DatasetSeTaSimulationRunningStatsAdapter


class TaskStatsAdapter(DatasetSeTaSimulationRunningStatsAdapter):
    """A general subject for the experiment, but this one logs also SE aux data."""

    @abstractmethod
    def get_task_id(self) -> float:
        pass

    @abstractmethod
    def get_task_instance_id(self) -> float:
        pass

    @abstractmethod
    def get_task_status(self) -> float:
        pass

    @abstractmethod
    def get_task_instance_status(self) -> float:
        pass

    @abstractmethod
    def get_reward(self) -> float:
        pass

    @abstractmethod
    def get_testing_phase(self) -> float:
        pass

    @abstractmethod
    def switch_learning(self, learning_on: bool):
        pass
