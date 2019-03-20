from abc import abstractmethod
from typing import Tuple, Dict, List, Union

from torchsim.research.experiment_templates.dataset_simulation_running_stats_template import \
    DatasetSeTaSimulationRunningStatsAdapter, DatasetSeSimulationRunningStatsExperimentTemplate

CONST_SEED = 3


class SeTaSimulationRunningStatsAdapter(DatasetSeTaSimulationRunningStatsAdapter):
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


class SeSimulationRunningStatsExperimentTemplate(DatasetSeSimulationRunningStatsExperimentTemplate):
    """The same as the base class, but this one also publishes SE aux data."""

    def __init__(self,
                 topology_adapter_instance: SeTaSimulationRunningStatsAdapter,
                 topology_class,
                 models_params: Union[List[Tuple], List[Dict]],
                 max_steps: int,
                 measurement_period: int = 5,
                 smoothing_window_size: int = 9,
                 seed=None,
                 experiment_name: str = 'empty_name',
                 save_cache=True,
                 load_cache=True,
                 clear_cache=True,
                 computation_only=False,
                 experiment_folder=None,
                 disable_plt_show=True):
        """Initializes the experiment testing the stability of learning a reasonably good representation of the MNIST digits.

        Args:
            topology_adapter_instance: subject of this experiment (contains the model and defines how to access its values)
            topology_class: class of the model to test
            models_params: parameters for each instance of model_class
            max_steps: number of steps for one run
            measurement_period: how many sim steps to wait between measurements
            smoothing_window_size: how many measurements to use for one evaluation (computation of the MI here)
            seed: seed for the controller
            disable_plt_show: disables blocking plt.show after end of measurement
        """
        super().__init__(
            topology_adapter_instance=topology_adapter_instance,
            topology_class=topology_class,
            models_params=models_params,
            max_steps=max_steps,
            measurement_period=measurement_period,
            smoothing_window_size=smoothing_window_size,
            seed=seed,
            experiment_name=experiment_name,
            save_cache=save_cache,
            load_cache=load_cache,
            clear_cache=clear_cache,
            computation_only=computation_only,
            experiment_folder=experiment_folder,
            disable_plt_show=disable_plt_show)

    def _register_se_aux_measurements(self):

        self._measurement_manager.add_measurement_f_with_period(
            'task_id',
            self._topology_adapter.get_task_id,
            self._measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'task_instance_id',
            self._topology_adapter.get_task_instance_id,
            self._measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'task_status',
            self._topology_adapter.get_task_status,
            self._measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'task_instance_status',
            self._topology_adapter.get_task_instance_status,
            self._measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'reward',
            self._topology_adapter.get_reward,
            self._measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'testing_phase',
            self._topology_adapter.get_testing_phase,
            self._measurement_period
        )

        self._task_ids = []
        self._task_instance_ids = []
        self._task_statuses = []
        self._task_instance_statuses = []
        self._rewards = []
        self._testing_phases = []

    def _append_list_to_aux_results(self):
        self._task_ids.append([])
        self._task_instance_ids.append([])
        self._task_statuses.append([])
        self._task_instance_statuses.append([])
        self._rewards.append([])
        self._testing_phases.append([])

    def _append_aux_results(self, measurement):
        self._task_ids[-1].append(measurement['task_id'])
        self._task_instance_ids[-1].append(measurement['task_instance_id'])
        self._task_statuses[-1].append(measurement['task_status'])
        self._task_instance_statuses[-1].append(measurement['task_instance_status'])
        self._rewards[-1].append(measurement['reward'])
        self._testing_phases[-1].append(measurement['testing_phase'])

    def _publish_aux_results(self, labels, date, docs_folder, doc):
        title = 'task id in ' + self._experiment_name
        self.plot_save(title, self._task_ids, 'Task id', labels, date, docs_folder, doc)

        title = 'task instance id in ' + self._experiment_name
        self.plot_save(title, self._task_instance_ids, 'Task instance id', labels, date, docs_folder, doc)

        title = 'task status in ' + self._experiment_name
        self.plot_save(title, self._task_statuses, 'Task status', labels, date, docs_folder, doc)

        title = 'task instance status in ' + self._experiment_name
        self.plot_save(title, self._task_instance_statuses, 'Task instance status', labels, date, docs_folder, doc)

        title = 'reward in ' + self._experiment_name
        self.plot_save(title, self._rewards, 'Reward', labels, date, docs_folder, doc)

        title = 'testing phase in ' + self._experiment_name
        self.plot_save(title, self._testing_phases, 'Testing phase', labels, date, docs_folder, doc)










