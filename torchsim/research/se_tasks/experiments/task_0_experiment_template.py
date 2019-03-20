from typing import Tuple, Dict, List, Union, Any

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval.doc_generator.heading import Heading
from torchsim.core.eval.experiment_template_base import TaskExperimentStatistics
from torchsim.research.experiment_templates.dataset_simulation_running_stats_template import \
    DatasetSeSimulationRunningStatsExperimentTemplate
from torchsim.research.se_tasks.adapters.task_stats_adapter import TaskStatsAdapter

CONST_SEED = 3


class Task0ExperimentTemplate(DatasetSeSimulationRunningStatsExperimentTemplate):
    """The same as the base class, but this one also publishes SE aux data."""

    def __init__(self,
                 topology_adapter_instance: TaskStatsAdapter,
                 topology_class,
                 models_params: List[Union[Tuple[Any], Dict[str, Any]]],
                 max_steps: int,
                 measurement_period: int = 1,
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

    def _should_end_run(self):
        last_measurements = self._measurement_manager.run_measurements[-1]
        return last_measurements.get_item('task_instance_id', last_measurements.get_last_step()) == -1

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
        self._tasks_solved = []
        self._instances_seen_train = []
        self._instances_seen_test = []
        self._instances_solved_train = []
        self._instances_solved_test = []

    def _append_list_to_aux_results(self):
        self._task_ids.append([])
        self._task_instance_ids.append([])
        self._task_statuses.append([])
        self._task_instance_statuses.append([])
        self._rewards.append([])
        self._testing_phases.append([])

    def _after_run_finished(self):
        super()._after_run_finished()

        last_measurements = self._measurement_manager.run_measurements[-1]  # get the last measurement (last run)

        instances_seen_train = 0
        instances_seen_test = 0
        instances_solved_train = 0
        instances_solved_test = 0

        tasks_solved = []

        for measurement in last_measurements:
            task_id = measurement['task_id']
            task_status = measurement['task_status']
            task_instance_status = measurement['task_instance_status']
            testing_phase = measurement['testing_phase']

            just_saw_new_task = False

            if len(tasks_solved) == 0 or tasks_solved[-1].task_id != task_id:
                tasks_solved.append(TaskExperimentStatistics(task_id))
                just_saw_new_task = True

            if task_status != 0:
                is_task_solved_successfully = task_status == 1
                task_index = -2 if just_saw_new_task else -1  # the task status for T-2 may get sent together with T-1
                tasks_solved[task_index].set_task_solved(is_task_solved_successfully)

            if task_instance_status != 0:
                is_instance_solved_successfully = task_instance_status == 1
                is_testing_phase = testing_phase == 1
                tasks_solved[-1].add_instance(is_instance_solved_successfully, is_testing_phase)

            self._task_ids[-1].append(measurement['task_id'])
            self._task_instance_ids[-1].append(measurement['task_instance_id'])
            self._task_statuses[-1].append(measurement['task_status'])
            self._task_instance_statuses[-1].append(measurement['task_instance_status'])
            self._rewards[-1].append(measurement['reward'])
            self._testing_phases[-1].append(measurement['testing_phase'])

        self._tasks_solved.append(tasks_solved)
        self._instances_seen_train.append(instances_seen_train)
        self._instances_seen_test.append(instances_seen_test)
        self._instances_solved_train.append(instances_solved_train)
        self._instances_solved_test.append(instances_solved_test)

    def _publish_aux_results(self, labels, date, docs_folder, doc):

        self._print_statistics(doc)

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

        smoothing = len(self._rewards[0])//100  # adaptive smoothing
        if smoothing % 2 == 0:
            smoothing += 1  # make odd
        title = f'reward (smoothing {smoothing}) in ' + self._experiment_name
        self.plot_save(title, self._rewards, 'Reward', labels, date, docs_folder, doc, smoothing_window_size=smoothing)

        title = 'testing phase in ' + self._experiment_name
        self.plot_save(title, self._testing_phases, 'Testing phase', labels, date, docs_folder, doc)

    def _print_statistics(self, doc: Document):
        doc.add(Heading("Overall statistics"))
        for run in range(len(self._tasks_solved)):
            doc.add(Heading(f"    Run {run}:", 2))
            experiment_statistics = self._tasks_solved[run]
            tasks_solved_statuses = [[stats.task_id, stats.task_solved] for stats in experiment_statistics]
            doc.add(f"Tasks solved: {tasks_solved_statuses}.<br />")
            for task_stats in experiment_statistics:
                self._print_task_stats(doc, task_stats)

    def _print_task_stats(self, doc: Document, task_stats: TaskExperimentStatistics):
        doc.add(Heading(f"Task {task_stats.task_id}<br />", 3))
        doc.add(f"Solved: {task_stats.task_solved}<br />")

        if task_stats.instances_seen_training != 0:
            ratio = task_stats.instances_solved_training / task_stats.instances_seen_training
        else:
            ratio = 0
        doc.add(f"Instances solved / seen during training: {task_stats.instances_solved_training} / "
                f"{task_stats.instances_seen_training}  = {ratio*100:.1f} %<br />")

        if task_stats.instances_seen_testing != 0:
            ratio = task_stats.instances_solved_testing / task_stats.instances_seen_testing
        else:
            ratio = 0
        doc.add(f"Instances solved / seen during testing: {task_stats.instances_solved_testing} / "
                f"{task_stats.instances_seen_testing}  = {ratio*100:.1f} %<br />")
