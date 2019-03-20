from os import path
import logging

import numpy as np
from abc import abstractmethod
from typing import Tuple, Dict, List, Union, Any

import torch

from torchsim.core.eval.metrics.simple_classifier_nn import compute_nn_classifier_accuracy
from torchsim.core.eval.topology_adapter_base import TopologyAdapterBase
from torchsim.core.eval.experiment_template_base import ExperimentTemplateBase
from torchsim.core.eval.metrics.mutual_information_metric import compute_mutual_information
from torchsim.core.eval.series_plotter import get_stamp, add_fig_to_doc, to_safe_name, \
    plot_multiple_runs_with_baselines, plot_multiple_runs
from torchsim.core.eval.doc_generator.document import Document

CONST_SEED = 3

logger = logging.getLogger(__name__)


class SpLearningConvergenceTopologyAdapter(TopologyAdapterBase):
    """A general subject for the experiment: SpLearningConvergenceExperiment."""

    @abstractmethod
    def get_label_id(self) -> int:
        """Returns an ID indicating current class label, from range <0, NO_CLASSES)."""
        pass

    @abstractmethod
    def get_learned_model_output_id(self) -> int:
        """Returns a scalar ID of the output of the learned model, from range <0, get_model_output_size)."""
        pass

    @abstractmethod
    def get_baseline_output_id(self) -> int:
        """Returns a scalar random ID from range <0, get_model_output_size)."""
        pass

    @abstractmethod
    def get_title(self) -> str:
        """Returns a description of the experiment for the report (plot)."""
        pass

    @abstractmethod
    def get_device(self) -> str:
        """Returns 'cpu' or 'cuda' based on the device."""
        pass

    @abstractmethod
    def get_model_output_size(self) -> int:
        """Return the dimension of the output (of the laerned model and the baseline).

        get_learned_model_output_id and get_baseline_output_id should be smaller than this value.
        """
        pass

    @abstractmethod
    def get_average_delta(self) -> float:
        pass

    @abstractmethod
    def get_average_boosting_duration(self) -> float:
        pass


class SpLearningConvergenceExperimentTemplate(ExperimentTemplateBase):
    """Sp learning convergence stability with respect to different initialization conditions."""

    _measurement_period: int
    _evaluation_period: int
    _sliding_window_stride: int

    _identical_model_initialization: bool
    _identical_mnist_initialization: bool

    _num_classes: int

    def __init__(self,
                 topology_adapter: SpLearningConvergenceTopologyAdapter,
                 topology_class,
                 models_params: List[Union[Tuple[Any], Dict[str, Any]]],
                 max_steps: int,
                 num_classes: int,
                 measurement_period: int = 5,
                 sliding_window_size: int = 9,
                 sliding_window_stride: int = 5,
                 sp_evaluation_period: int = 10,
                 seed=None,
                 experiment_name: str = 'empty_name',
                 save_cache=True,
                 load_cache=True,
                 clear_cache=True,
                 computation_only=False,
                 experiment_folder=None,
                 disable_plt_show=True,
                 debug_mi=False):
        """Initializes the experiment testing the stability of learning a reasonably good representation of the MNIST digits.

        Args:
            topology_adapter: subject of this experiment (contains the model and defines how to access its values)
            topology_class: class of the model to test
            models_params: parameters for each instance of model_class
            max_steps: number of steps for one run
            measurement_period: how many sim steps to wait between measurements
            sliding_window_size: how many measurements to use for one evaluation (computation of the MI here)
            seed: seed for the controller
        """
        super().__init__(topology_adapter, topology_class, models_params, seed=seed,
                         max_steps=max_steps,
                         save_cache=save_cache,
                         load_cache=load_cache,
                         computation_only=computation_only,
                         disable_plt_show=disable_plt_show,
                         experiment_folder=experiment_folder,
                         clear_cache=clear_cache,
                         experiment_name=experiment_name)

        self._sliding_window_stride = sliding_window_stride
        self._sp_evaluation_period = sp_evaluation_period

        self._topology_adapter = topology_adapter
        self._measurement_period = measurement_period
        self._evaluation_period = sliding_window_size

        self._num_classes = num_classes
        self._topology_class = topology_class

        self._debug_mi = debug_mi

        # collect labels of the dataset and outputs of the model with a given period
        self._measurement_manager = self._create_measurement_manager(self._experiment_folder, delete_after_each_run=True)
        self._measurement_manager.add_measurement_f_with_period(
            'dataset_labels',
            self._topology_adapter.get_label_id,
            self._measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'model_outputs',
            self._topology_adapter.get_learned_model_output_id,
            self._measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'baseline_outputs',
            self._topology_adapter.get_baseline_output_id,
            self._measurement_period
        )

        self._measurement_manager.add_measurement_f_with_period(
            'average_delta',
            self._topology_adapter.get_average_delta,
            self._sp_evaluation_period
        )

        self._measurement_manager.add_measurement_f_with_period(
            'average_boosting_dur',
            self._topology_adapter.get_average_boosting_duration,
            self._sp_evaluation_period
        )

        self._mutual_info = []
        self._baseline_mutual_info = []

        self._classifier_accuracy = []
        self._baseline_classifier_accuracy = []

        self._average_boosting_dur = []
        self._average_delta = []

        self._steps = []
        self._different_steps = []

    def _after_run_finished(self):
        """Called after each run (one model with one set of parameters) is finished.

        Measurement manager collected multiple values during the run, we can process them here
        and store the results locally.

        I.e. here compute the mutual_info, baseline_mutual info from a sliding window of labels/outputs.
        """

        # append lists for this run
        self._mutual_info.append([])
        self._baseline_mutual_info.append([])

        self._classifier_accuracy.append([])
        self._baseline_classifier_accuracy.append([])

        self._steps.append([])

        self._average_boosting_dur.append([])
        self._average_delta.append([])
        self._different_steps.append([])

        # get all the measurements from the last run
        last_run_measurements = self._measurement_manager.run_measurements[-1]

        # temporary sliding window for computation of one value of the mutual information
        labels_window = []
        outputs_window = []
        baseline_outputs_window = []

        print('computing statistics after run...')

        # go step-by-step through the last run (single_measurement contains all the values taken in that time-step)
        for single_measurement in last_run_measurements:

            # these two measurements have to run with different (lower) frequency
            if 'average_boosting_dur' in single_measurement.keys():
                self._average_boosting_dur[-1].append(single_measurement['average_boosting_dur'])
                self._average_delta[-1].append(single_measurement['average_delta'])
                self._different_steps[-1].append(single_measurement['current_step'])

            # pick "dataset_labels" (see the init()) from the single_measurement and append one value to the separate list
            labels_window.append(single_measurement['dataset_labels'])
            outputs_window.append(single_measurement['model_outputs'])
            baseline_outputs_window.append(single_measurement['baseline_outputs'])

            # wait until the window has enough values
            if len(labels_window) < self._evaluation_period:
                continue

            # compute stats in the window and store to the last run (that's the [-1]) at the end (that's the append)
            self._mutual_info[-1].append(
                compute_mutual_information(
                    np.array(labels_window),
                    np.array(outputs_window),
                    self._num_classes,
                    data_contains_id=True)
            )

            if self._debug_mi:
                self._debug_mutual_info(np.array(labels_window), np.array(outputs_window), self._mutual_info[-1][-1])

            self._baseline_mutual_info[-1].append(
                compute_mutual_information(
                    np.array(labels_window),
                    np.array(baseline_outputs_window),
                    self._num_classes,
                    data_contains_id=True)
            )

            # compute the classifier accuracies (for model and baseline)
            dev = self._topology_adapter.get_device()
            output_dim = self._topology_adapter.get_model_output_size()

            labels_tensor = torch.tensor(labels_window, dtype=torch.long, device=dev)
            outputs_tensor = torch.tensor(outputs_window, dtype=torch.long, device=dev)
            baseline_outputs_tensor = torch.tensor(baseline_outputs_window, dtype=torch.long, device=dev)

            acc = self._compute_classifier_acc(outputs_tensor, labels_tensor, output_dim)
            self._classifier_accuracy[-1].append(acc)

            baseline_acc = self._compute_classifier_acc(baseline_outputs_tensor, labels_tensor, output_dim)
            self._baseline_classifier_accuracy[-1].append(baseline_acc)

            # store also step (for the x-axis)
            self._steps[-1].append(single_measurement['current_step'])

            # remove the self._sliding_window_stride items from the sliding windows.. (then fill the same amount..)
            for i in range(0, self._sliding_window_stride):
                if len(labels_window) > 0:
                    labels_window.pop(0)
                    outputs_window.pop(0)
                    baseline_outputs_window.pop(0)

    @staticmethod
    def _representation_perfect(label_to_output: Dict[int, set], output_to_label: Dict[int, set]) -> bool:

        for source, targets in label_to_output.items():
            if len(targets) != 1:
                return False

        for source, targets in output_to_label.items():
            if len(targets) != 1:
                return False

        return True

    def _debug_mutual_info(self, labels: np.array, outputs: np.array, mi: float):
        """Test the mutual info metric.

         Use in case there should be 1to1 correspondence of cluster centers and data points
         (n_cluster_centers = n_data_points). In this case, the mutual info should be 1.0
        """
        print(f'MI={mi} between \nl\t[{",".join(map(str, labels))}] and \no\t[{",".join(map(str, outputs))}]')

        label_to_output = {}
        output_to_label = {}

        for cls in range(0, self._num_classes):
            label_to_output[cls] = set()
            output_to_label[cls] = set()

        for label, output in zip(labels, outputs):
            label_to_output[label].add(output)
            output_to_label[output].add(label)

        logger.debug(f'label->output: {label_to_output}')
        logger.debug(f'output->label: {output_to_label}')

        if SpLearningConvergenceExperimentTemplate._representation_perfect(label_to_output, output_to_label):
            logger.debug(f'representation is perfect, mutual info should be 1 and is {mi}')
            if mi > 1.000001 or mi < 0.99999:
                logger.error(f'Mutual info is {mi} but should be 1!')

    def _compute_classifier_acc(self, outputs: torch.Tensor, labels: torch.Tensor, output_dim: int) -> float:
        """Compute the classifier accuracy trained on the outputs to predict the labels.

         A single output has a scalar format from the range <0,output_dim),
         A single label has a scalar format from the range <0,n_classes).

        Args:
            outputs: [n_samples]
            labels: [n_samples]
            output_dim: range of ids in the outputs tensor

        Returns:
            Low-VC dimension classifier accuracy trained and tested on this data.
        """
        acc = compute_nn_classifier_accuracy(outputs,
                                             labels,
                                             n_classes=self._num_classes,
                                             custom_input_length=output_dim,
                                             log_loss=False,
                                             max_epochs=50)
        return acc

    def _compute_experiment_statistics(self):
        """Called after all runs are finished (have available all the measurements in the self._measurement_manager)."""
        pass

    def _get_measurement_manager(self):
        return self._measurement_manager

    def _experiment_template_name(self):
        return self._topology_adapter.__class__.__name__

    def plot_save(self, title, series, series_baselines, ylabel, labels, date, docs_folder, doc,
                  smoothing_size: int = None):
        """Baselines are in dotted grey, the model outputs are colored."""

        f = plot_multiple_runs_with_baselines(
            self._steps[0],
            series,
            series_baselines,
            title=title,
            xlabel='Simulation step',
            ylim=[0, 1.1],
            ylabel=ylabel,
            smoothing_window_size=smoothing_size,
            labels=labels)

        add_fig_to_doc(f, path.join(docs_folder, title), doc)

    def _complete_name(self):
        return f"{self._topology_class.__name__}_" + self._experiment_name + "_"

    def _publish_results(self):
        """Plot and save the results."""

        doc = Document()
        date = get_stamp()

        labels = ExperimentTemplateBase.parameters_to_string(self._topology_parameters_list)

        title = 'Mutual Information labels vs ' + self._experiment_name
        self.plot_save(title,
                       self._mutual_info,
                       self._baseline_mutual_info,
                       'Norm. mutual information',
                       labels, date, self._docs_folder, doc)

        title = 'Weak classifier accuracy labels vs ' + self._experiment_name
        self.plot_save(title,
                       self._classifier_accuracy,
                       self._baseline_classifier_accuracy,
                       'Classifier accuracy',
                       labels, date, self._docs_folder, doc)  #, smoothing_size=3)

        title = 'average delta'
        f = plot_multiple_runs(
            self._different_steps[0],  # here the X axes are identical
            self._average_delta,
            title=title,
            ylabel='log(delta)',
            xlabel='steps',
            labels=labels
        )
        add_fig_to_doc(f, path.join(self._docs_folder, title), doc)

        title = 'average boosting duration'
        f = plot_multiple_runs(
            self._different_steps[0],
            self._average_boosting_dur,
            title=title,
            ylabel='duration',
            xlabel='steps',
            labels=labels
        )
        add_fig_to_doc(f, path.join(self._docs_folder, title), doc)

        doc.write_file(path.join(self._docs_folder, to_safe_name(self._complete_name() + date + ".html")))

        print('done')



