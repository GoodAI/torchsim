import logging
import os
from abc import abstractmethod
from functools import partial
from os import path
from typing import List, Tuple, Union, Dict

import torch

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval.experiment_template_base import ExperimentTemplateBase
from torchsim.core.eval.measurement_manager import MeasurementManagerBase
from torchsim.core.eval.series_plotter import plot_multiple_runs, add_fig_to_doc, plot_multiple_runs_with_baselines
from torchsim.research.experiment_templates.task0_train_test_template_base import Task0TrainTestTemplateBase, \
    Task0TrainTestTemplateAdapterBase
from torchsim.utils.template_utils.template_helpers import compute_classification_accuracy, \
    compute_se_classification_accuracy, compute_label_reconstruction_accuracies, compute_mse_values

logger = logging.getLogger(__name__)


class Task0TrainTestClassificationAccAdapter(Task0TrainTestTemplateAdapterBase):
    """Provide access to all layers of the topology, provide functionality to switch train/test etc."""

    @abstractmethod
    def get_average_log_delta_for(self, layer_id: int) -> float:
        """Average log delta across all dimensions and all the cluster centers. If delta==0, return 0."""
        pass

    @abstractmethod
    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        """Returns a tensor representing the class label predicted by the architecture."""
        pass

    @abstractmethod
    def is_learning(self) -> bool:
        """Return true if learning"""
        pass


class Task0TrainTestClassificationAccTemplate(Task0TrainTestTemplateBase):
    """An experiment which measures classification accuracy on the Task0 and compares with the baseline"""

    _measurement_period: int
    _sliding_window_size: int
    _sliding_window_stride: int

    _num_classes: int
    _num_layers: int

    _layer_measurement_managers: List

    def __init__(self,
                 topology_adapter: Task0TrainTestClassificationAccAdapter,
                 topology_class,
                 topology_parameters: Union[List[Tuple], List[Dict]],
                 num_layers: int,
                 num_classes: int,
                 overall_training_steps: int,
                 num_testing_steps: int,
                 num_testing_phases: int,
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
                 disable_plt_show=True):
        super().__init__(topology_adapter=topology_adapter,
                         topology_class=topology_class,
                         sp_evaluation_period=sp_evaluation_period,
                         measurement_period=measurement_period,
                         topology_parameters=topology_parameters,
                         sliding_window_stride=sliding_window_stride,
                         sliding_window_size=sliding_window_size,
                         experiment_name=experiment_name,
                         overall_training_steps=overall_training_steps,
                         num_testing_steps=num_testing_steps,
                         num_testing_phases=num_testing_phases,
                         save_cache=save_cache,
                         load_cache=load_cache,
                         clear_cache=clear_cache,
                         computation_only=computation_only,
                         seed=seed,
                         disable_plt_show=disable_plt_show,
                         experiment_folder=experiment_folder
                         )

        self._num_layers = num_layers
        self._num_classes = num_classes

        # value available even during testing
        self._measurement_manager.add_measurement_f_with_period_testing(
            'dataset_label_ids',
            topology_adapter.get_label_id,
            self._measurement_period
        )

        # available during testing
        self._measurement_manager.add_measurement_f_with_period_testing(
            'dataset_tensors',
            topology_adapter.clone_ground_truth_label_tensor,
            self._measurement_period
        )

        # constant zeros, for MSE baseline
        self._measurement_manager.add_measurement_f_with_period_testing(
            'zeros_baseline_tensor',
            topology_adapter.clone_constant_baseline_output_tensor_for_labels,
            self._measurement_period
        )

        # random one-hot vectors
        self._measurement_manager.add_measurement_f_with_period_testing(
            'random_baseline_tensor',
            topology_adapter.clone_random_baseline_output_tensor_for_labels,
            self._measurement_period
        )

        # output of the learned model (TA/NNet)
        self._measurement_manager.add_measurement_f_with_period_testing(
            'predicted_label_tensor',
            topology_adapter.clone_predicted_label_tensor_output,
            self._measurement_period
        )

        # average deltas (just to visually check that the model converges?)
        delta_getter = partial(topology_adapter.get_average_log_delta_for, 0)
        self._measurement_manager.add_measurement_f_with_period_training(
            'average_delta0_train',
            delta_getter,
            self._sp_evaluation_period
        )
        delta_getter_1 = partial(topology_adapter.get_average_log_delta_for, 1)
        self._measurement_manager.add_measurement_f_with_period_training(
            'average_delta1_train',
            delta_getter_1,
            self._sp_evaluation_period
        )

        # mostly for debug purposes: learning should be off during testing.
        self._measurement_manager.add_measurement_f_with_period(
            'is_learning',
            topology_adapter.is_learning,
            self._measurement_period
        )

        self._steps = []
        self._plotted_testing_phase_id = []
        self._plotted_training_phase_id = []

        self._test_sp_steps = []
        self._average_delta_train_0 = []
        self._average_delta_train_1 = []

        self._predicted_labels_mse = []  # mse between labels and their predictions
        self._baseline_labels_mse = []  # mse between labels and random number outputs

        self._model_accuracy = []
        self._baseline_accuracy = []

        self._model_se_accuracy = []
        self._baseline_se_accuracy = []

        self._is_learning_info = []

    def _get_measurement_manager(self) -> MeasurementManagerBase:
        return self._measurement_manager

    def _experiment_template_name(self):
        return self._experiment_name

    def _after_run_finished(self):

        last_run_measurements = self._measurement_manager.run_measurements[-1]

        print('computing statistics after run...')

        self._steps.append([])

        self._plotted_testing_phase_id.append([])
        self._plotted_training_phase_id.append([])

        self._test_sp_steps.append([])
        self._average_delta_train_0.append([])
        self._average_delta_train_1.append([])

        self._is_learning_info.append([])

        # --------- deltas
        for single_measurement in last_run_measurements:
            self._is_learning_info[-1].append(single_measurement['is_learning'])
            self._steps[-1].append(single_measurement['current_step'])
            self._plotted_testing_phase_id[-1].append(single_measurement['testing_phase_id'])
            self._plotted_training_phase_id[-1].append(single_measurement['training_phase_id'])

            if 'average_delta0_train' in single_measurement:
                self._average_delta_train_0[-1].append(single_measurement['average_delta0_train'])
                self._average_delta_train_1[-1].append(single_measurement['average_delta1_train'])
                self._test_sp_steps[-1].append(single_measurement['current_step'])

        # ---------- MSE
        baseline_zeros_output = last_run_measurements.partition_to_list_of_testing_phases('zeros_baseline_tensor')
        model_classification_outputs = last_run_measurements.partition_to_list_of_testing_phases(
            'predicted_label_tensor')
        dataset_tensors = last_run_measurements.partition_to_list_of_testing_phases('dataset_tensors')
        baseline_mse = compute_mse_values(dataset_tensors, baseline_zeros_output)
        model_mse = compute_mse_values(dataset_tensors, model_classification_outputs)
        self._baseline_labels_mse.append(baseline_mse)
        self._predicted_labels_mse.append(model_mse)

        # ----------- Classification accuracy
        ground_truth_ids = last_run_measurements.partition_to_list_of_testing_phases('dataset_label_ids')
        random_outputs = last_run_measurements.partition_to_list_of_testing_phases('random_baseline_tensor')

        base_acc, model_acc = compute_label_reconstruction_accuracies(
            ground_truth_ids=ground_truth_ids,
            baseline_output_tensors=random_outputs,
            model_output_tensors=model_classification_outputs,
            accuracy_method=compute_classification_accuracy,
            num_classes=self._num_classes
        )
        self._baseline_accuracy.append(base_acc)
        self._model_accuracy.append(model_acc)

        base_se_acc, model_se_acc = compute_label_reconstruction_accuracies(
            ground_truth_ids=ground_truth_ids,
            baseline_output_tensors=random_outputs,
            model_output_tensors=model_classification_outputs,
            accuracy_method=compute_se_classification_accuracy,
            num_classes=self._num_classes
        )
        self._baseline_se_accuracy.append(base_se_acc)
        self._model_se_accuracy.append(model_se_acc)

        print('done')

    def _compute_experiment_statistics(self):
        pass

    def _publish_results_to_doc(self, doc: Document, date: str, docs_folder: os.path):
        """An alternative to the _publish_results method, this is called from _publish_results now

        Draw and add your topologies to the document here.
        """

        labels = ExperimentTemplateBase.extract_params_for_legend(self._topology_parameters_list)

        title = 'training_phase_id'
        f = plot_multiple_runs(
            self._steps,
            self._plotted_training_phase_id,
            title=title,
            ylabel='training_phase_id',
            xlabel='steps',
            labels=labels
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)

        title = 'testing_phase_id'
        f = plot_multiple_runs(
            self._steps,
            self._plotted_testing_phase_id,
            title=title,
            ylabel='testing_phase_id',
            xlabel='steps',
            labels=labels
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)

        # plot the classification accuracy
        title = 'Is Learning'
        f = plot_multiple_runs(
            self._steps,
            self._is_learning_info,
            title=title,
            ylabel='learning=True?',
            xlabel='steps',
            ylim=[-0.1, 1.1],
            labels=labels
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)

        max_x = max(self._steps[0])

        title = 'average_delta_train_layer0'
        f = plot_multiple_runs(
            self._test_sp_steps[0],
            self._average_delta_train_0,
            title=title,
            ylabel='average_deltas',
            xlabel='steps',
            labels=labels,
            xlim=[0, max_x],
            disable_ascii_labels=True,
            hide_legend=True
            # use_scatter=True
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)

        title = 'average_delta_train_layer1'
        f = plot_multiple_runs(
            self._test_sp_steps[0],
            self._average_delta_train_1,
            title=title,
            ylabel='average_deltas',
            xlabel='steps',
            labels=labels,
            xlim=[0, max_x],
            disable_ascii_labels=True,
            hide_legend=True
            # use_scatter=True
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)

        testing_phases_x = list(range(0, len(self._predicted_labels_mse[0])))

        # plot the classification accuracy
        title = 'Label reconstruction accuracy (step-wise)'
        f = plot_multiple_runs_with_baselines(
            testing_phases_x,
            self._model_accuracy,
            self._baseline_accuracy,
            title=title,
            ylabel='accuracy (1 ~ 100%)',
            xlabel='testing phase ID',
            ylim=[-0.1, 1.1],
            labels=labels,
            hide_legend=True
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)

        # plot the classification accuracy
        title = 'Label reconstruction SE accuracy (step-wise)'
        f = plot_multiple_runs_with_baselines(
            testing_phases_x,
            self._model_se_accuracy,
            self._baseline_se_accuracy,
            title=title,
            ylabel='accuracy (1 ~ 100%)',
            xlabel='testing_phase ID',
            ylim=[-0.1, 1.1],
            labels=labels,
            hide_legend=True
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)

        # plot the MSE
        title = 'Mean Square Error of label reconstruction'
        f = plot_multiple_runs_with_baselines(
            testing_phases_x,
            self._predicted_labels_mse,
            self._baseline_labels_mse,
            title=title,
            ylabel='MSE',
            xlabel='testing phase ID',
            labels=labels,
            hide_legend=True,
            ylim=[-0.1, 0.2]  # just for better resolution
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)

