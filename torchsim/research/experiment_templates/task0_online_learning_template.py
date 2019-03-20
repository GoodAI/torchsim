from os import path
import logging

import numpy as np
from abc import abstractmethod, ABC
from typing import Tuple, Dict, List, Union, Any

import torch

from torchsim.core.eval.measurement_manager import MeasurementManager
from torchsim.core.eval.topology_adapter_base import TopologyAdapterBase
from torchsim.core.eval.experiment_template_base import ExperimentTemplateBase
from torchsim.core.eval.metrics.mutual_information_metric import compute_mutual_information
from torchsim.core.eval.series_plotter import get_stamp, add_fig_to_doc, to_safe_name, \
    plot_multiple_runs_with_baselines, plot_multiple_runs
from torchsim.core.eval.doc_generator.document import Document

from functools import partial
from torchsim.research.research_topics.rt_1_1_4_task0_experiments.adapters.task0_adapter_base import Task0AdapterBase

from torchsim.utils.template_utils.template_helpers import do_compute_nn_classifier_accuracy, compute_mse_from, \
    compute_classification_accuracy, compute_se_classification_accuracy, argmax_tensor

CONST_SEED = 3

logger = logging.getLogger(__name__)


class Task0LearningAdapterBase(TopologyAdapterBase, ABC):
    """Common adapter for the online and train/test splitted experiment measurement."""

    @abstractmethod
    def get_label_id(self) -> int:
        """Should return ID indicating current class label, from range <0, NO_CLASSES)."""
        pass

    @abstractmethod
    def clone_label_tensor(self) -> torch.Tensor:
        """Returns a tensor containing the current label."""
        pass

    @abstractmethod
    def clone_ground_truth_label_tensor(self) -> torch.Tensor:
        """Returns a tensor containing the current label, but not hidden during testing."""
        pass

    @abstractmethod
    def clone_baseline_output_tensor_for_labels(self) -> torch.Tensor:
        """Returns a tensor with length of num_labels, representing prediction of class label from random baseline."""
        pass

    @abstractmethod
    def clone_random_baseline_output_tensor_for_labels(self) -> torch.Tensor:
        """The get_baseline_output_tensor_for_labels is from constant baseline, this one is random baseline."""
        pass

    @abstractmethod
    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        """Returns a tensor representing the class label predicted by the architecture."""
        pass

    @abstractmethod
    def get_baseline_output_id_for(self, layer_id: int) -> int:
        """Returns a scalar, random ID from range <0, get_model_output_size)."""
        pass

    @abstractmethod
    def get_sp_output_size_for(self, layer_id: int) -> int:
        """Returns the dimension of the output (of the learned model and the baseline).

        get_learned_model_output_id and get_baseline_output_id should be smaller than this value.
        """
        pass

    @abstractmethod
    def clone_sp_output_tensor_for(self, layer_id: int) -> torch.Tensor:
        """SP output of a given layer (used for simple classifier for flocksize >=1)."""
        pass

    @abstractmethod
    def get_sp_output_id_for(self, layer_id: int) -> int:
        """Returns an ID corresponding to the output of the learned model, from range <0, get_model_output_size)."""
        pass

    @abstractmethod
    def is_output_id_available_for(self, layer_id: int) -> bool:
        """Layers with flock_size > 1 cannot provide id (just the output tensor)."""
        pass

    @abstractmethod
    def get_average_log_delta_for(self, layer_id: int) -> float:
        pass

    @abstractmethod
    def get_average_boosting_duration_for(self, layer_id: int) -> float:
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
    def get_current_step(self) -> int:
        """Returns the current step of the topology."""
        pass


class Task0OnlineLearningAdapterBase(Task0LearningAdapterBase):
    """This adapted is used by this template."""

    @abstractmethod
    def switch_learning(self, learning_on: bool):
        """Turn the learning of all the nodes on/off."""
        pass

    @abstractmethod
    def dataset_switch_learning(self, learning_on: bool, just_hide_labels: bool):
        """Turn the learning on/of in the dataset (SE do not support this command)."""
        pass

    @abstractmethod
    def is_learning(self) -> bool:
        pass


class Task0OnlineLearningTemplate(ExperimentTemplateBase):
    """Usefulness of representation of various topologies on the Task0, measured online."""

    _measurement_period: int
    _sliding_window_size: int
    _sliding_window_stride: int

    _identical_model_initialization: bool
    _identical_mnist_initialization: bool

    _num_classes: int
    _num_layers: int

    _layer_measurement_managers: List

    _num_training_steps: int

    _just_hide_labels: bool

    def __init__(self,
                 topology_adapter: Task0AdapterBase,
                 topology_class,
                 models_params: List[Union[Tuple[Any], Dict[str, Any]]],
                 max_steps: int,
                 num_training_steps: int,
                 num_classes: int,
                 num_layers: int,
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
                 just_hide_labels=False):
        """Template for runtime evaluation of performance of the topology on M1T0.

        Args:
            topology_adapter: unified way for accessing potentially different topologies
            topology_class: topology to be evaluated
            models_params: list of params to be used in the topology init
            max_steps: how many steps to run the measurement
            num_training_steps: after this number of steps the testing phase is initiated
            num_classes: number of classes
            num_layers: number of layers of the topology
            measurement_period: how often (in simulation steps) to measure things
            sliding_window_size: values are computed from the sliding window of this size (in measurements)k
            sliding_window_stride: skip this amount of measurements
            sp_evaluation_period: how often (in simulation steps) to compute SP internal stats (everage delta, boosting)
            seed: probably not used here
            experiment_name:
            save_cache: cache measurements
            load_cache: load measurements if available
            clear_cache: clear
            computation_only: disable measurement, load
            experiment_folder: where to save the results (automatically done well)
            disable_plt_show: disables showing topologies (currently overriden also eslewhere..)
            just_hide_labels: when switching to the testing phase (in SEObjects datast), we can either hide the labels
            and jump to testing data, or just hide the labels and continue
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

        self._just_hide_labels = just_hide_labels
        self._num_training_steps = num_training_steps
        self._sliding_window_stride = sliding_window_stride
        self._sp_evaluation_period = sp_evaluation_period

        self._topology_adapter = topology_adapter
        self._measurement_period = measurement_period
        self._sliding_window_size = sliding_window_size

        self._num_classes = num_classes
        self._topology_class = topology_class
        self._num_layers = num_layers

        # collect labels of the dataset and outputs of the model with a given period
        self._measurement_manager = self._create_measurement_manager(self._experiment_folder, delete_after_each_run=True)

        self._layer_measurement_managers = []

        # each layer manager handles registering the measurements and computing the statistics for own layer
        # common statistics remain in this class
        for layer_id in range(0, self._num_layers):
            manager = Task0LayerMeasurementManager(layer_id=layer_id,
                                                   experiment_name=experiment_name,
                                                   num_classes=self._num_classes,
                                                   measurement_manager=self._get_measurement_manager(),
                                                   topology_adapter=self._topology_adapter,
                                                   sp_evaluation_period=self._sp_evaluation_period,
                                                   measurement_period=self._measurement_period,
                                                   sliding_window_stride=self._sliding_window_stride,
                                                   sliding_window_size=sliding_window_size)

            self._layer_measurement_managers.append(manager)

        for manager in self._layer_measurement_managers:
            manager.register_measurements()

        # value available even during testing
        self._measurement_manager.add_measurement_f_with_period(
            'dataset_labels',
            self._topology_adapter.get_label_id,
            self._measurement_period
        )

        # available during testing
        self._measurement_manager.add_measurement_f_with_period(
            'dataset_tensors',
            self._topology_adapter.clone_ground_truth_label_tensor,
            self._measurement_period
        )

        # constant zeros, for MSE
        self._measurement_manager.add_measurement_f_with_period(
            'label_baseline_tensor',
            self._topology_adapter.clone_baseline_output_tensor_for_labels,
            self._measurement_period
        )

        # random IDs, for classification accuracy baseline
        self._measurement_manager.add_measurement_f_with_period(
            'random_label_baseline_id',
            self._topology_adapter.get_random_baseline_output_id_for_labels,
            self._measurement_period
        )

        # output of the TA
        self._measurement_manager.add_measurement_f_with_period(
            'predicted_label_tensor',
            self._topology_adapter.clone_predicted_label_tensor_output,
            self._measurement_period
        )

        self._measurement_manager.add_measurement_f_with_period(
            'is_learning',
            self._topology_adapter.is_learning,
            self._measurement_period
        )

        self._steps = []
        self._predicted_labels_mse = []  # mse between labels and their predictions
        self._baseline_labels_mse = []  # mse between labels and random number outputs
        self._is_learning_info = []
        self._classification_accuracy = []  # accuracy of classification in the window
        self._random_classification_accuracy = []

        self._se_classification_accuracy = []  # accuracy computed in the same way as in SE
        self._se_random_classification_accuracy = []  # .. answer is the most frequent argmax(output) for one object

    def _after_run_finished(self):
        """Called after each run (one model with one set of parameters) is finished.

        Measurement manager collected multiple values during the run, we can process them here
        and store the results locally.

        I.e. here compute the mutual_info, baseline_mutual info from a sliding window of labels/outputs.
        """

        for manager in self._layer_measurement_managers:
            manager.prepare_after_run_finished()

        last_run_measurements = self._measurement_manager.run_measurements[-1]

        print('computing statistics after run...')

        self._steps.append([])
        self._predicted_labels_mse.append([])
        self._baseline_labels_mse.append([])
        self._is_learning_info.append([])
        self._classification_accuracy.append([])
        self._random_classification_accuracy.append([])
        self._se_classification_accuracy.append([])
        self._se_random_classification_accuracy.append([])

        labels_window = []
        predictions_window = []
        baseline_window = []

        random_label_ids = []
        predicted_label_ids = []
        correct_ids = []

        # go step-by-step through the last run (single_measurement contains all the values taken in that time-step)
        for single_measurement in last_run_measurements:

            for manager in self._layer_measurement_managers:
                manager.compute_single_measurement_stats(single_measurement)

            # common stats here
            labels_window.append(single_measurement['dataset_tensors'])
            predictions_window.append(single_measurement['predicted_label_tensor'])
            baseline_window.append(single_measurement['label_baseline_tensor'])

            random_label_ids.append(single_measurement['random_label_baseline_id'])
            predicted_label_ids.append(argmax_tensor(single_measurement['predicted_label_tensor']))
            correct_ids.append(single_measurement['dataset_labels'])

            # wait until the window has enough values
            if len(labels_window) < self._sliding_window_size:
                continue

            # statistics computed from sliding window here:
            base_mse = compute_mse_from(labels_window, baseline_window)
            model_mse = compute_mse_from(labels_window, predictions_window)

            random_accuracy = compute_classification_accuracy(correct_ids, random_label_ids)
            model_accuracy = compute_classification_accuracy(correct_ids, predicted_label_ids)

            se_random_accuracy = compute_se_classification_accuracy(correct_ids, random_label_ids, self._num_classes)
            se_model_accuracy = compute_se_classification_accuracy(correct_ids, predicted_label_ids, self._num_classes)

            self._steps[-1].append(single_measurement['current_step'])
            self._predicted_labels_mse[-1].append(model_mse)
            self._baseline_labels_mse[-1].append(base_mse)
            self._is_learning_info[-1].append(int(single_measurement['is_learning']))

            self._classification_accuracy[-1].append(model_accuracy)
            self._random_classification_accuracy[-1].append(random_accuracy)

            self._se_random_classification_accuracy[-1].append(se_random_accuracy)
            self._se_classification_accuracy[-1].append(se_model_accuracy)

            # remove the self._sliding_window_stride items from the sliding windows.. (then fill the same amount..)
            for _ in range(0, self._sliding_window_stride):
                if len(labels_window) > 0:
                    labels_window.pop(0)
                    predictions_window.pop(0)
                    baseline_window.pop(0)
                    random_label_ids.pop(0)
                    predicted_label_ids.pop(0)
                    correct_ids.pop(0)

    def _compute_experiment_statistics(self):
        """Called after all runs are finished (have available all the measurements in the self._measurement_manager)."""
        pass

    def _get_measurement_manager(self):
        return self._measurement_manager

    def _experiment_template_name(self):
        return self._topology_adapter.__class__.__name__

    @staticmethod
    def plot_save(title, steps, series, series_baselines, ylabel, labels, date, docs_folder, doc,
                  smoothing_size: int = None):
        """Baselines are in dotted grey, the model outputs are colored."""

        f = plot_multiple_runs_with_baselines(
            np.array(steps),
            np.array(series),
            np.array(series_baselines),
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

        for manager in self._layer_measurement_managers:
            manager.publish_results(labels=labels, date=date, docs_folder=self._docs_folder, doc=doc)

        # plot the running MSE
        title = 'Mean Square Error of TA classification'
        f = plot_multiple_runs_with_baselines(
            self._steps,
            self._predicted_labels_mse,
            self._baseline_labels_mse,
            title=title,
            ylabel='mse',
            xlabel='steps',
            labels=labels
        )
        add_fig_to_doc(f, path.join(self._docs_folder, title), doc)

        title = 'TA classification accuracy'
        f = plot_multiple_runs_with_baselines(
            self._steps,
            self._classification_accuracy,
            self._random_classification_accuracy,
            title=title,
            ylabel='accuracy (1 ~ 100%)',
            xlabel='steps',
            labels=labels
        )
        add_fig_to_doc(f, path.join(self._docs_folder, title), doc)

        title = 'TA classification accuracy - (SE-metric)'
        f = plot_multiple_runs_with_baselines(
            self._steps,
            self._se_classification_accuracy,
            self._se_random_classification_accuracy,
            title=title,
            ylabel='accuracy (1 ~ 100%)',
            xlabel='steps',
            labels=labels
        )
        add_fig_to_doc(f, path.join(self._docs_folder, title), doc)

        title = 'is_learning'
        f = plot_multiple_runs(
            self._steps,
            self._is_learning_info,
            title=title,
            ylabel='is learning',
            xlabel='steps',
        )
        add_fig_to_doc(f, path.join(self._docs_folder, title), doc)

        doc.write_file(path.join(self._docs_folder, to_safe_name(self._complete_name() + date + ".html")))

        print('done')

    def _do_before_topology_step(self):
        """Turn on testing after given number of steps."""
        super()._do_before_topology_step()

        should_be_testing = self._topology_adapter.get_current_step() > self._num_training_steps

        if self._topology_adapter.is_learning() is should_be_testing:
            self._topology_adapter.switch_learning(learning_on=(not should_be_testing))
            self._topology_adapter.dataset_switch_learning(learning_on=(not should_be_testing),
                                                           just_hide_labels=self._just_hide_labels)


class Task0LayerMeasurementManager:
    """Some experiments might need collecting the same statistics for multiple layers.

    This collects common statistics for each layer (one instance for one layer).
    """

    _layer_id: int
    _num_classes: int  # num classes of the dataset
    _measurement_manager: MeasurementManager
    _topology_adapter: Task0OnlineLearningAdapterBase
    _sliding_window_stride: int
    _sliding_window_size: int
    _sp_evaluation_period: int
    _measurement_period: int
    _experiment_name: str

    _steps: List
    _mutual_info: List
    _baseline_mutual_info: List
    _classifier_accuracy: List
    _baseline_classifier_accuracy: List

    _different_steps: List
    _average_boosting_dur: List
    _average_delta: List

    _labels_window = []
    _outputs_window = []
    _baseline_outputs_window = []

    def __init__(self, layer_id: int,
                 measurement_manager: MeasurementManager,
                 topology_adapter: Task0OnlineLearningAdapterBase,
                 num_classes: int,
                 sp_evaluation_period: int,
                 measurement_period: int,
                 sliding_window_stride: int,
                 sliding_window_size: int,
                 experiment_name: str
                 ):
        self._layer_id = layer_id
        self._num_classes = num_classes
        self._measurement_manager = measurement_manager
        self._topology_adapter = topology_adapter
        self._sliding_window_stride = sliding_window_stride
        self._sliding_window_size = sliding_window_size
        self._sp_evaluation_period = sp_evaluation_period
        self._measurement_period = measurement_period
        self._experiment_name = experiment_name

    def _id_to_str(self):
        return str(self._layer_id)

    def _is_id_available(self):
        return self._topology_adapter.is_output_id_available_for(self._layer_id)

    def register_measurements(self):

        if self._is_id_available():
            output_getter = partial(self._topology_adapter.get_sp_output_id_for, self._layer_id)
            self._measurement_manager.add_measurement_f_with_period(
                'model_outputs' + self._id_to_str(),
                output_getter,
                self._measurement_period
            )

        baseline_zero_getter = partial(self._topology_adapter.get_baseline_output_id_for, self._layer_id)
        self._measurement_manager.add_measurement_f_with_period(
            'baseline_outputs' + self._id_to_str(),
            baseline_zero_getter,
            self._measurement_period
        )

        delta_getter = partial(self._topology_adapter.get_average_log_delta_for, self._layer_id)
        self._measurement_manager.add_measurement_f_with_period(
            'average_delta' + self._id_to_str(),
            delta_getter,
            self._sp_evaluation_period
        )

        boosting_dur_getter = partial(self._topology_adapter.get_average_boosting_duration_for, self._layer_id)
        self._measurement_manager.add_measurement_f_with_period(
            'average_boosting_dur' + self._id_to_str(),
            boosting_dur_getter,
            self._sp_evaluation_period
        )

        # init the lists
        self._steps = []
        self._mutual_info = []
        self._baseline_mutual_info = []
        self._classifier_accuracy = []
        self._baseline_classifier_accuracy = []

        self._different_steps = []
        self._average_boosting_dur = []
        self._average_delta = []

    def prepare_after_run_finished(self):

        # append results for new measurement
        self._steps.append([])
        self._mutual_info.append([])
        self._baseline_mutual_info.append([])
        self._classifier_accuracy.append([])
        self._baseline_classifier_accuracy.append([])

        self._different_steps.append([])
        self._average_boosting_dur.append([])
        self._average_delta.append([])

        # clear temp windows
        self._labels_window = []
        self._outputs_window = []
        self._baseline_outputs_window = []

    def compute_single_measurement_stats(self, single_measurement):
        # these two measurements have to run with different (lower) frequency
        if 'average_boosting_dur' + self._id_to_str() in single_measurement.keys():
            self._average_boosting_dur[-1].append(single_measurement['average_boosting_dur' + self._id_to_str()])
            self._average_delta[-1].append(single_measurement['average_delta' + self._id_to_str()])
            self._different_steps[-1].append(single_measurement['current_step'])

        # pick "dataset_labels" (see the init()) from the single_measurement and append one value to the separate list
        self._labels_window.append(single_measurement['dataset_labels'])  # TODO move back to the base
        if self._is_id_available():
            self._outputs_window.append(single_measurement['model_outputs' + self._id_to_str()])
        else:
            self._outputs_window.append(0.01)
        self._baseline_outputs_window.append(single_measurement['baseline_outputs' + self._id_to_str()])

        # wait until the window has enough values
        if len(self._labels_window) < self._sliding_window_size:
            return

        if self._is_id_available():
            # compute stats in the window and store to the last run (that's the [-1]) at the end (that's the append)
            self._mutual_info[-1].append(
                compute_mutual_information(
                    np.array(self._labels_window),
                    np.array(self._outputs_window),
                    self._num_classes,
                    data_contains_id=True)
            )
        else:
            self._mutual_info[-1].append(0.0)

        self._baseline_mutual_info[-1].append(
            compute_mutual_information(
                np.array(self._labels_window),
                np.array(self._baseline_outputs_window),
                self._num_classes,
                data_contains_id=True)
        )

        # compute the classifier accuracies (for model and baseline)
        dev = self._topology_adapter.get_device()
        output_dim = self._topology_adapter.get_sp_output_size_for(self._layer_id)

        # classifier accuracy on the outputs of the layer
        # TODO could be made that the classifier computes from tensors even for the layers with flock_size>1
        if self._is_id_available():
            acc = do_compute_nn_classifier_accuracy(self._outputs_window,
                                                    self._labels_window,
                                                    output_dim,
                                                    self._num_classes,
                                                    device=dev)
            self._classifier_accuracy[-1].append(acc)
        else:
            # add dummy values
            self._classifier_accuracy[-1].append(0.0)

        baseline_acc = do_compute_nn_classifier_accuracy(self._baseline_outputs_window,
                                                         self._labels_window,
                                                         output_dim,
                                                         self._num_classes,
                                                         device=dev)
        self._baseline_classifier_accuracy[-1].append(baseline_acc)

        # store also step (for the x-axis)
        self._steps[-1].append(single_measurement['current_step'])

        # remove the self._sliding_window_stride items from the sliding windows.. (then fill the same amount..)
        for i in range(0, self._sliding_window_stride):
            if len(self._labels_window) > 0:
                self._labels_window.pop(0)
                self._outputs_window.pop(0)
                self._baseline_outputs_window.pop(0)

    def _available(self):
        if self._is_id_available():
            return ''
        return '[NA] '

    def publish_results(self, labels, date, docs_folder, doc):

        prefix = 'L' + str(self._layer_id) + '--'

        title = prefix + self._available() + 'Mutual Information labels vs ' + self._experiment_name
        Task0OnlineLearningTemplate.plot_save(title,
                                              self._steps,
                                              self._mutual_info,
                                              self._baseline_mutual_info,
                                              'Norm. mutual information',
                                              labels, date, docs_folder, doc)

        title = prefix + self._available() + 'Weak classifier accuracy labels vs ' + self._experiment_name
        Task0OnlineLearningTemplate.plot_save(title,
                                              self._steps,
                                              self._classifier_accuracy,
                                              self._baseline_classifier_accuracy,
                                              'Classifier accuracy',
                                              labels, date, docs_folder, doc)  # , smoothing_size=3)

        # TODO why this is different than the two above?
        title = prefix + 'average delta'
        f = plot_multiple_runs(
            self._different_steps,
            self._average_delta,
            title=title,
            ylabel='log(delta)',
            xlabel='steps',
            labels=labels
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)

        title = prefix + 'average boosting duration'
        f = plot_multiple_runs(
            self._different_steps,
            self._average_boosting_dur,
            title=title,
            ylabel='duration',
            xlabel='steps',
            labels=labels
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)
