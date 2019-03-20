import logging
from abc import ABC, abstractmethod
from functools import partial
from os import path
from typing import List, Tuple, Union, Dict, Any

import torch
from torchsim.core.eval.doc_generator.document import Document

from torchsim.core.eval.experiment_template_base import ExperimentTemplateBase
from torchsim.core.eval.measurement_manager import MeasurementManagerBase
from torchsim.core.eval.metrics.mutual_information_metric import compute_mutual_information_for_phases
from torchsim.core.eval.metrics.simple_classifier_nn import NNClassifier
from torchsim.core.eval.series_plotter import plot_multiple_runs, add_fig_to_doc, plot_multiple_runs_with_baselines
from torchsim.core.eval.testable_measurement_manager import TestableMeasurementManager
from torchsim.research.experiment_templates.task0_train_test_classification_acc_template import \
    Task0TrainTestClassificationAccAdapter
from torchsim.research.experiment_templates.task0_train_test_template_base import Task0TrainTestTemplateBase
from torchsim.research.research_topics.rt_2_1_2_learning_rate.utils.cluster_agreement_measurement import \
    ClusterAgreementMeasurement
from torchsim.utils.template_utils.template_helpers import argmax_list_list_tensors, partition_to_list_of_ids, \
    compute_mse_values, \
    compute_label_reconstruction_accuracies, compute_classification_accuracy, compute_se_classification_accuracy

logger = logging.getLogger(__name__)


class TaTask0TrainTestClassificationAccAdapter(Task0TrainTestClassificationAccAdapter, ABC):
    """Supports measurement of everything"""

    def is_learning(self) -> bool:
        return self.is_in_training_phase()

    @abstractmethod
    def get_output_id_for(self, layer_id: int) -> int:
        pass

    @abstractmethod
    def clone_sp_output_tensor_for(self, layer_id: int) -> int:
        pass

    @abstractmethod
    def get_average_boosting_duration_for(self, layer_id: int) -> float:
        pass

    @abstractmethod
    def get_num_boosted_clusters_ratio(self, layer_id: int) -> float:
        """Return number of boosted clusters in the layers divided by the total boosted clusters"""
        pass

    @abstractmethod
    def get_sp_size_for(self, layer_id: int) -> int:
        pass

    @abstractmethod
    def get_flock_size_of(self, layer_id: int) -> int:
        pass


class Task0TrainTestLearningRateTemplate(Task0TrainTestTemplateBase):
    _measurement_period: int
    _sliding_window_size: int
    _sliding_window_stride: int

    _plot_is_learning: bool

    _num_classes: int
    _num_layers: int

    # result of the measurement, used for unit testing, contains only results of the LAST run!
    _test_deltas_filtered: List[Tuple[int, List[Tuple[int, Any]]]]
    _train_deltas_filtered: List[Tuple[int, List[Tuple[int, Any]]]]

    _topology_adapter: TaTask0TrainTestClassificationAccAdapter

    _layer_measurement_managers: List['Task0TrainTestLayerMeasurementManager']

    def __init__(self,
                 topology_adapter: TaTask0TrainTestClassificationAccAdapter,
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
                 disable_plt_show=True,
                 show_conv_agreements=True,
                 plot_is_learning=False):
        """ Compared to the base, this also computes more TA-specific statistics.

        For example: cluster_agreements for each run, boosting duration etc..
        """
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

        self._topology_adapter = topology_adapter
        self._show_conv_agreements = show_conv_agreements
        self._plot_is_learning = plot_is_learning

        # register layer measurement managers
        self._layer_measurement_managers = []

        for layer_id in range(0, self._num_layers):
            manager = Task0TrainTestLayerMeasurementManager(
                layer_id=layer_id,
                measurement_manager=self._measurement_manager,
                num_classes=num_classes,
                num_layers=num_layers,
                sp_evaluation_period=sp_evaluation_period,
                measurement_period=measurement_period,
                experiment_name=experiment_name,
                topology_parameters_list=self._topology_parameters_list,
                topology_adapter=self._topology_adapter,
                show_conv_agreements=self._show_conv_agreements
            )
            self._layer_measurement_managers.append(manager)

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

        # mostly for debug purposes: learning should be off during testing.
        self._measurement_manager.add_measurement_f_with_period(
            'is_learning',
            topology_adapter.is_learning,
            self._measurement_period
        )

        # read output ids for the top-level expert
        output_id_getter = partial(topology_adapter.get_output_id_for, self._top_layer_id())
        self._measurement_manager.add_measurement_f_with_period_testing(
            'output_id_' + str(self._top_layer_id()),
            output_id_getter,
            self._measurement_period
        )
        self._is_learning_info = []

        # top-layer stats
        self._mutual_info = []
        self._base_mutual_info = []

        self._steps = []
        self._plotted_testing_phase_id = []
        self._plotted_training_phase_id = []

        self._test_sp_steps = []

        self._predicted_labels_mse = []  # mse between labels and their predictions
        self._baseline_labels_mse = []  # mse between labels and random number outputs

        self._model_accuracy = []
        self._baseline_accuracy = []

        self._model_se_accuracy = []
        self._baseline_se_accuracy = []

    def _get_measurement_manager(self) -> MeasurementManagerBase:
        return self._measurement_manager

    def _experiment_template_name(self):
        return self._experiment_name

    def _top_layer_id(self):
        return self._num_layers - 1

    def _after_run_finished(self):
        super()._after_run_finished()

        print('Computing common statistics...')
        self._is_learning_info.append([])
        self._steps.append([])
        self._plotted_testing_phase_id.append([])
        self._plotted_training_phase_id.append([])
        self._test_sp_steps.append([])

        last_run_measurements = self._measurement_manager.run_measurements[-1]

        for single_measurement in last_run_measurements:
            self._is_learning_info[-1].append(single_measurement['is_learning'])
            self._steps[-1].append(single_measurement['current_step'])
            self._plotted_testing_phase_id[-1].append(single_measurement['testing_phase_id'])
            self._plotted_training_phase_id[-1].append(single_measurement['training_phase_id'])

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

        # clustering agreement on the top layer
        output_ids = last_run_measurements.partition_to_list_of_testing_phases('output_id_' + str(self._top_layer_id()))

        # mutual info on L1
        random_output_ids = argmax_list_list_tensors(random_outputs)

        base_mi = compute_mutual_information_for_phases(
            labels=ground_truth_ids,
            data=random_output_ids,
            num_classes=self._num_classes,
            data_contains_id=True)

        l1_mi = compute_mutual_information_for_phases(
            labels=ground_truth_ids,
            data=output_ids,
            num_classes=self._num_classes,
            data_contains_id=True
        )
        self._base_mutual_info.append(base_mi)
        self._mutual_info.append(l1_mi)

        print('Computing layer-wise statistics...')

        # compute the layer-wise stats
        for manager in self._layer_measurement_managers:
            manager.after_run_finished()

        print('Done')

    def _compute_experiment_statistics(self):
        pass

    def _publish_results_to_doc(self, doc: Document, date: str, docs_folder: path):
        """Adds my results to the results produced by the base class"""
        super()._publish_results_to_doc(doc, date, docs_folder)

        doc.add(f"<br><br><br><b>Common results</b><br>")

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

        if self._plot_is_learning:
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

        testing_phase_ids = list(range(0, len(self._mutual_info[0])))  # x-axis values

        title = 'Top-layer L' + str(self._top_layer_id()) + ' Mutual information of SP output with labels'
        f = plot_multiple_runs_with_baselines(
            testing_phase_ids,
            self._mutual_info,
            self._base_mutual_info,
            title=title,
            ylabel='Normalized mutual information',
            xlabel='steps',
            labels=labels,
            ylim=[-0.1, 1.1],
            hide_legend=True
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)

        # plot the classification accuracy
        title = 'Label reconstruction accuracy (step-wise)'
        f = plot_multiple_runs_with_baselines(
            testing_phase_ids,
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
        title = 'Label reconstruction SE accuracy'
        f = plot_multiple_runs_with_baselines(
            testing_phase_ids,
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
            testing_phase_ids,
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

        for manager in reversed(self._layer_measurement_managers):
            manager.publish_results(labels=title, date=date, docs_folder=docs_folder, doc=doc)


class Task0TrainTestLayerMeasurementManager:
    """Some experiments might need collecting the same statistics for multiple layers.

    This collects common statistics for each layer (one instance for one layer).
    """

    _layer_id: int
    _num_classes: int  # num classes of the dataset
    _num_layers: int
    _measurement_manager: TestableMeasurementManager
    _topology_adapter: TaTask0TrainTestClassificationAccAdapter
    _sp_evaluation_period: int
    _measurement_period: int
    _experiment_name: str
    _show_conv_agreements: bool

    def __init__(self, layer_id: int,
                 measurement_manager: TestableMeasurementManager,
                 topology_adapter: TaTask0TrainTestClassificationAccAdapter,
                 num_classes: int,
                 num_layers: int,  # an assumption is that all layers are conv except the last one
                 sp_evaluation_period: int,
                 measurement_period: int,
                 experiment_name: str,
                 topology_parameters_list,
                 show_conv_agreements: bool
                 ):
        self._layer_id = layer_id
        self._num_classes = num_classes
        self._num_layers = num_layers
        self._measurement_manager = measurement_manager
        self._topology_adapter = topology_adapter
        self._sp_evaluation_period = sp_evaluation_period
        self._measurement_period = measurement_period
        self._experiment_name = experiment_name
        self._topology_parameters_list = topology_parameters_list
        self._show_conv_agreements = show_conv_agreements

        # read output tensors for convolutional flock
        output_tensor_getter = partial(topology_adapter.clone_sp_output_tensor_for, self._layer_id)
        self._measurement_manager.add_measurement_f_with_period_testing(
            'output_tensors_' + self._id_to_str(),
            output_tensor_getter,
            self._measurement_period
        )

        boosting_dur_getter = partial(self._topology_adapter.get_average_boosting_duration_for, self._layer_id)
        self._measurement_manager.add_measurement_f_with_period_training(
            'average_boosting_dur_' + self._id_to_str(),
            boosting_dur_getter,
            self._sp_evaluation_period
        )

        num_boosted_getter = partial(self._topology_adapter.get_num_boosted_clusters_ratio, self._layer_id)
        self._measurement_manager.add_measurement_f_with_period_training(
            'num_boosted_getter_' + self._id_to_str(),
            num_boosted_getter,
            self._sp_evaluation_period
        )

        delta_getter = partial(topology_adapter.get_average_log_delta_for, self._layer_id)
        self._measurement_manager.add_measurement_f_with_period(
            'average_delta_' + self._id_to_str(),
            delta_getter,
            self._sp_evaluation_period
        )

        self._clustering_agreements = []  # List (experiment_run), List(expert_id), List(phase), List(meas..)

        self._sp_evaluation_steps = []
        self._average_boosting_dur = []
        self._num_boosted_clusters = []
        self._average_deltas = []

        self._weak_class_accuracy = []
        self._base_weak_class_accuracy = []

    def _id_to_str(self):
        return str(self._layer_id)

    def after_run_finished(self):

        self._average_boosting_dur.append([])
        self._num_boosted_clusters.append([])
        self._sp_evaluation_steps.append([])
        self._average_deltas.append([])

        last_run_measurements = self._measurement_manager.run_measurements[-1]

        for single_measurement in last_run_measurements:
            if 'average_boosting_dur_' + self._id_to_str() in single_measurement:
                self._average_boosting_dur[-1].append(single_measurement['average_boosting_dur_' + self._id_to_str()])
                self._num_boosted_clusters[-1].append(single_measurement['num_boosted_getter_' + self._id_to_str()])
                self._sp_evaluation_steps[-1].append(single_measurement['current_step'])
                self._average_deltas[-1].append(single_measurement['average_delta_' + self._id_to_str()])

        # ---------- Cluster agreement
        output_tensors = last_run_measurements.partition_to_list_of_testing_phases('output_tensors_' + self._id_to_str())
        partitioned = partition_to_list_of_ids(output_tensors,
                                               self._topology_adapter.get_flock_size_of(self._layer_id))

        # go through each expert and compute its agreements
        all_agreements = []
        for expert_output_ids in partitioned:
            agreements = ClusterAgreementMeasurement.compute_cluster_agreements(expert_output_ids,
                                                                                compute_self_agreement=True)
            all_agreements.append(agreements)
        self._clustering_agreements.append(all_agreements)

        # ---------- Weak classifier accuracy on the top layer
        ground_truth_ids = last_run_measurements.partition_to_list_of_testing_phases('dataset_label_ids')
        random_outputs = last_run_measurements.partition_to_list_of_testing_phases('random_baseline_tensor')
        random_output_ids = argmax_list_list_tensors(random_outputs)

        output_tensors_reshaped = self.flatten_output_tensors(output_tensors)

        output_size = self._topology_adapter.get_sp_size_for(self._layer_id)

        classifier = NNClassifier()
        model_acc = classifier.train_and_evaluate_in_phases(output_tensors_reshaped,
                                                            ground_truth_ids,
                                                            # classifier_input_size=output_size,
                                                            n_classes=self._num_classes,
                                                            device='cuda')  # TODO derive device from the topology

        base_acc = classifier.train_and_evaluate_in_phases(random_output_ids,
                                                           ground_truth_ids,
                                                           classifier_input_size=output_size,
                                                           n_classes=self._num_classes,
                                                           device='cuda')
        self._weak_class_accuracy.append(model_acc)
        self._base_weak_class_accuracy.append(base_acc)

    @staticmethod
    def flatten_output_tensors(output_tensors: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        """ output tensors have shape [flock_size, output_size], reshape to [flock_size * output_size]
        """
        result = []
        for phase in output_tensors:
            result.append([])
            for measurement in phase:
                result[-1].append(measurement.view(-1))
        return result

    def publish_results(self, labels, date, docs_folder, doc):

        doc.add(f"<br><br><br><b>Results for layer {self._layer_id}</b><br>")

        prefix = 'L' + str(self._layer_id) + '--'

        testing_phase_ids = list(range(0, len(self._clustering_agreements[0][0])))  # x-axis values

        title = prefix + ' Weak classifier accuracy trained on SP outputs to labels'
        f = plot_multiple_runs_with_baselines(
            testing_phase_ids,
            self._weak_class_accuracy,
            self._base_weak_class_accuracy,
            title=title,
            ylabel='Accuracy (1 ~ 100%)',
            xlabel='steps',
            labels=labels,
            ylim=[-0.1, 1.1],
            hide_legend=True
        )

        add_fig_to_doc(f, path.join(docs_folder, title), doc)

        title = prefix + '- Average boosting duration'
        f = plot_multiple_runs(
            self._sp_evaluation_steps[0],
            self._average_boosting_dur,
            title=title,
            ylabel='duration',
            xlabel='steps',
            labels=labels,
            hide_legend=True
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)

        title = prefix + '- Num boosted clusters'
        f = plot_multiple_runs(
            self._sp_evaluation_steps[0],
            self._num_boosted_clusters,
            title=title,
            ylabel='Num boosted clusters / total clusters',
            xlabel='steps',
            labels=labels,
            ylim=[-0.1, 1.1],
            hide_legend=True
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)

        title = prefix + '- Average_deltas'
        f = plot_multiple_runs(
            self._sp_evaluation_steps[0],
            self._average_deltas,
            title=title,
            ylabel='average_deltas',
            xlabel='steps',
            labels=labels,
            disable_ascii_labels=True,
            hide_legend=True
            # use_scatter=True
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)

        # if this is not the top layer, show conv agreements only if required
        if self._show_conv_agreements or self._layer_id == (self._num_layers - 1):
            agreements = self._clustering_agreements

            for run_id, run_agreements in enumerate(agreements):
                for expert_id, expert_agreements in enumerate(run_agreements):
                    self._plot_agreement(
                        prefix,
                        expert_id,
                        run_id,
                        testing_phase_ids,
                        expert_agreements,
                        docs_folder,
                        doc
                    )

    def _plot_agreement(self,
                        prefix: str,
                        expert_id: int,
                        run_id: int,
                        testing_phase_ids: List[int],
                        agreements: List[List[int]],
                        docs_folder,
                        doc):
        """Plot cluster agreements for one expert in a layer"""
        run_params = ExperimentTemplateBase.parameters_to_string([self._topology_parameters_list[run_id]])

        title = prefix + f'- E{expert_id}-Run{run_id} - Clustering agreements'
        f = plot_multiple_runs(
            testing_phase_ids,
            agreements,
            title=title,
            ylabel='agreement',
            # xlabel=f'params: {run_params}',
            xlabel=f'testing phases',
            disable_ascii_labels=True,
            hide_legend=True,
            ylim=[ClusterAgreementMeasurement.NO_VALUE - 0.1, 1.1]
        )
        add_fig_to_doc(f, path.join(docs_folder, title), doc)
