from datetime import datetime
from abc import abstractmethod
from os import path
from typing import Tuple, Dict, List, Union
import matplotlib.pyplot as plt

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval.experiment_template_base import ExperimentTemplateBase
from torchsim.core.eval.series_plotter import plot_multiple_runs, get_stamp, add_fig_to_doc, to_safe_name
from torchsim.core.eval.topology_adapter_base import TopologyAdapterBase

CONST_SEED = 3


class DatasetSeTaSimulationRunningStatsAdapter(TopologyAdapterBase):
    """A general subject for the experiment: SpLearningConvergenceExperiment."""

    @abstractmethod
    def get_max_memory_allocated(self) -> float:
        pass

    @abstractmethod
    def get_memory_allocated(self) -> float:
        pass

    @abstractmethod
    def get_max_memory_cached(self) -> float:
        pass

    @abstractmethod
    def get_title(self) -> str:
        """Return description of the experiment for the report (plot)."""
        pass

    @abstractmethod
    def get_memory_cached(self) -> float:
        pass


class DatasetSeSimulationRunningStatsExperimentTemplate(ExperimentTemplateBase):
    """Measures fps and memory requirements for different configurations of the model."""

    _measurement_period: int
    _smoothing_window_size: int

    _identical_model_initialization: bool
    _identical_mnist_initialization: bool

    _experiment_name: str

    def __init__(self,
                 topology_adapter_instance: DatasetSeTaSimulationRunningStatsAdapter,
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
        super().__init__(topology_adapter_instance, topology_class, models_params, seed=seed,
                         max_steps=max_steps,
                         save_cache=save_cache,
                         load_cache=load_cache,
                         computation_only=computation_only,
                         disable_plt_show=disable_plt_show,
                         experiment_name=experiment_name,
                         experiment_folder=experiment_folder,
                         clear_cache=clear_cache)

        self._topology_adapter = topology_adapter_instance
        self._measurement_period = measurement_period
        self._smoothing_window_size = smoothing_window_size

        # collect labels from the task and outputs of the model with a given period
        self._measurement_manager = self._create_measurement_manager(self._experiment_folder, delete_after_each_run=True)
        self._measurement_manager.add_measurement_f_with_period(
            'step_lengths',
            self._get_last_step_length,
            self._measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'current_memory',
            self._topology_adapter.get_memory_allocated,
            self._measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'max_memory',
            self._topology_adapter.get_max_memory_allocated,
            self._measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'current_cache',
            self._topology_adapter.get_memory_cached,
            self._measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'max_cache',
            self._topology_adapter.get_max_memory_cached,
            self._measurement_period
        )

        self._register_se_aux_measurements()

        self._fps = []
        self._current_mem = []
        self._max_mem = []
        self._max_cached = []
        self._current_cached = []
        self._steps = []

    def _get_last_step_length(self):
        diff = self.time_step_ended - self.time_step_started
        return diff.microseconds / 1000000.0 + diff.seconds

    def _do_before_topology_step(self):
        self.time_step_started = datetime.now()

    def _do_after_topology_step(self):
        self.time_step_ended = datetime.now()

    def _register_se_aux_measurements(self):
        pass

    def _append_list_to_results(self):
        # append list for this run
        self._steps.append([])
        self._current_mem.append([])
        self._fps.append([])
        self._max_mem.append([])
        self._max_cached.append([])
        self._current_cached.append([])

    def _append_list_to_aux_results(self):
        pass

    def _append_aux_results(self, measurement):
        pass

    def _after_run_finished(self):
        self._append_list_to_results()
        self._append_list_to_aux_results()

        last_measurements = self._measurement_manager.run_measurements[-1]  # get the last measurement (last run)

        # go through the measured values and append each at the end of the list
        for measurement in last_measurements:
            value = measurement['step_lengths']
            fps = 1.0 / (1 if value == 0 else value)
            self._fps[-1].append(fps)
            self._current_mem[-1].append(measurement['current_memory'])
            self._max_mem[-1].append(measurement['max_memory'])
            self._current_cached[-1].append(measurement['current_cache'])
            self._max_cached[-1].append(measurement['max_cache'])
            self._steps[-1].append(measurement['current_step'])

            self._append_aux_results(measurement)  # if something other is collected here

        print('done')

    def _compute_experiment_statistics(self):
        """Compute statistics from the collected measurements (moved to _after_run_finished())."""

    def _get_measurement_manager(self):
        return self._measurement_manager

    def _experiment_template_name(self):
        return self._topology_adapter.__class__.__name__

    def plot_save(self, title, series, ylabel, labels, date, docs_folder, doc, smoothing_window_size=None):
        f = plot_multiple_runs(
            self._steps,
            series,
            title=title,
            xlabel='Simulation step',
            ylabel=ylabel,
            labels=labels,
            smoothing_window_size=smoothing_window_size)
        add_fig_to_doc(f, path.join(docs_folder, to_safe_name(title)), doc)
        plt.close(f)

    def _publish_aux_results(self, labels, date, docs_folder, doc):
        pass

    def _publish_results(self):
        """Plot and optionally save the results."""
        doc = Document()
        date = get_stamp()

        labels = ExperimentTemplateBase.parameters_to_string(self._topology_parameters_list)

        title = f'FPS vs. {self._experiment_name} (smoothing {self._smoothing_window_size})'
        self.plot_save(title, self._fps, 'FPS',
                       labels, date, self._docs_folder, doc, smoothing_window_size=self._smoothing_window_size)

        smoothing = len(self._fps[0]) // 100  # adaptive smoothing
        if smoothing % 2 == 0:
            smoothing += 1  # make odd
        title = 'FPS vs. ' + self._experiment_name + f' (smoothing {smoothing})'
        self.plot_save(title, self._fps, 'FPS', labels, date, self._docs_folder, doc,
                       smoothing_window_size=smoothing)

        title = f'max_memory_allocated() vs. {self._experiment_name} (smoothing {self._smoothing_window_size})'
        self.plot_save(title, self._max_mem, 'Max memory', labels, date, self._docs_folder, doc,
                       smoothing_window_size=self._smoothing_window_size)

        title = f'memory_allocated() vs. {self._experiment_name} (smoothing {self._smoothing_window_size})'
        self.plot_save(title, self._current_mem, 'Current memory', labels, date, self._docs_folder, doc,
                       smoothing_window_size=self._smoothing_window_size)

        title = f'max_cached() vs. {self._experiment_name} (smoothing {self._smoothing_window_size})'
        self.plot_save(title, self._max_cached, 'Max cached mem', labels, date, self._docs_folder, doc,
                       smoothing_window_size=self._smoothing_window_size)

        title = f'memory_cached() vs. {self._experiment_name} (smoothing {self._smoothing_window_size})'
        self.plot_save(title, self._current_cached, 'Current cached mem', labels, date, self._docs_folder, doc,
                       smoothing_window_size=self._smoothing_window_size)

        self._publish_aux_results(labels, date, self._docs_folder, doc)

        doc.write_file(path.join(self._docs_folder, f"{self._topology_class.__name__}_" + date + ".html"))
        print('done')












