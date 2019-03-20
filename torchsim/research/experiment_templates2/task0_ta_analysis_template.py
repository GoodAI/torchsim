import logging
from functools import partial
from os import path
from typing import List

import torch

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval.metrics.mutual_information_metric import compute_mutual_information_for_phases
from torchsim.core.eval.metrics.simple_classifier_nn import NNClassifier
from torchsim.core.eval.metrics.simple_classifier_svm import SvmClassifier
from torchsim.core.eval.series_plotter import plot_multiple_runs, plot_multiple_runs_with_baselines
from torchsim.core.eval2.experiment_controller import ExperimentController, TrainTestComponentParams, ExperimentComponent, \
    TrainTestMeasuringComponent, TrainTestControllingComponent
from torchsim.core.eval2.experiment_template_base import ExperimentTemplateBase
from torchsim.core.eval2.measurement_manager import RunMeasurementManager, MeasurementManager
from torchsim.core.eval2.run_measurement import TrainTestMeasurementPartitioning
from torchsim.research.research_topics.rt_2_1_2_learning_rate.utils.cluster_agreement_measurement import \
    ClusterAgreementMeasurement
from torchsim.research.research_topics.rt_3_7_1_task0_analysis.topologies.task0_ta_analysis_topology import \
    Task0TaAnalysisTopology
from torchsim.utils.template_utils.template_helpers import compute_classification_accuracy, \
    compute_se_classification_accuracy, compute_label_reconstruction_accuracies, compute_mse_values, \
    argmax_list_list_tensors, partition_to_list_of_ids

logger = logging.getLogger(__name__)


class Task0TaAnalysisParams:
    measurement_period: int
    sp_evaluation_period: int
    show_conv_agreements: bool
    is_train_test_classifier_computed: bool

    num_layers: int
    num_classes: int

    def __init__(self,
                 measurement_period: int,
                 num_classes: int,
                 num_layers: int,
                 sp_evaluation_period: int,
                 show_conv_agreements: bool = False,
                 is_train_test_classifier_computed: bool = True):
        self.measurement_period = measurement_period
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.sp_evaluation_period = sp_evaluation_period
        self.show_conv_agreements = show_conv_agreements
        self.is_train_test_classifier_computed = is_train_test_classifier_computed


class Task0TaAnalysisLayerComponent(ExperimentComponent):
    _num_classes: int
    _num_layers: int
    _is_train_test_class_computed: bool

    _top_layer_id: int
    _layer_id: int

    def __init__(self,
                 topology: Task0TaAnalysisTopology,
                 run_measurement_manager: RunMeasurementManager,
                 experiment_params: Task0TaAnalysisParams,
                 train_test_component: TrainTestMeasuringComponent,
                 num_layers: int,
                 num_classes: int,
                 layer_id: int,
                 is_train_test_class_computed):

        self._is_train_test_class_computed = is_train_test_class_computed
        self._run_measurement_manager = run_measurement_manager
        self._train_test_component = train_test_component
        self._params = experiment_params
        self._topology = topology

        self._num_layers = num_layers
        self._num_classes = num_classes
        self._top_layer_id = num_layers - 1
        self._layer_id = layer_id

        # read output tensors for convolutional flock
        output_tensor_getter = partial(topology.model.clone_sp_output_tensor_for, self._layer_id)
        self._run_measurement_manager.add_measurement_f(
            f'output_tensors_{self._layer_id}',
            output_tensor_getter,
            self._params.measurement_period
        )

        boosting_dur_getter = partial(topology.model.get_average_boosting_duration_for, self._layer_id)
        self._train_test_component.add_measurement_f_training(
            f'average_boosting_dur_{self._layer_id}',
            boosting_dur_getter,
            self._params.sp_evaluation_period
        )

        num_boosted_getter = partial(topology.model.get_num_boosted_clusters_ratio, self._layer_id)
        self._train_test_component.add_measurement_f_training(
            f'num_boosted_getter_{self._layer_id}',
            num_boosted_getter,
            self._params.sp_evaluation_period
        )

        delta_getter = partial(topology.model.get_average_log_delta_for, self._layer_id)
        self._train_test_component.add_measurement_f_training(
            f'average_delta_{self._layer_id}',
            delta_getter,
            self._params.sp_evaluation_period
        )

    def calculate_run_results(self):
        super().calculate_run_results()

        measurements = self._run_measurement_manager.measurements
        logger.info(f'computing statistics for layer {self._layer_id} ...')

        train_test_partitioning = TrainTestMeasurementPartitioning(measurements)

        # ---------- Cluster agreement
        output_tensors = train_test_partitioning.partition_to_list_of_testing_phases(
            f'output_tensors_{self._layer_id}')
        partitioned = partition_to_list_of_ids(output_tensors,
                                               self._topology.model.get_flock_size_of(self._layer_id))

        logger.info('computing cluster agreements')
        # go through each expert and compute its agreements
        all_agreements = []
        for expert_output_ids in partitioned:
            agreements = ClusterAgreementMeasurement.compute_cluster_agreements(expert_output_ids,
                                                                                compute_self_agreement=True)
            all_agreements.append(agreements)

        measurements.add_custom_data(f'clustering_agreements_{self._layer_id}', all_agreements)

        # ---------- Weak classifier accuracy on the top layer
        ground_truth_ids = train_test_partitioning.partition_to_list_of_testing_phases('dataset_label_ids')
        # Note: all layer outputs are compared to random one-hot vectors of size num_classes
        # should be improved that baseline for each layer is flock_size*one-hot vectors of length num_cc
        random_outputs: List[List[torch.Tensor]] = train_test_partitioning.partition_to_list_of_testing_phases(
            'random_baseline_tensor')
        random_output_ids = argmax_list_list_tensors(random_outputs)
        output_size = random_outputs[0][0].numel()

        output_tensors_reshaped = self.flatten_output_tensors(output_tensors)

        classifier = SvmClassifier()

        if self._is_train_test_class_computed:
            # get also the training data
            ground_truth_ids_train = train_test_partitioning.partition_to_list_of_training_phases('dataset_label_ids')
            output_tensors_train = train_test_partitioning.partition_to_list_of_training_phases(
                f'output_tensors_{self._layer_id}')
            output_tensors_reshaped_train = self.flatten_output_tensors(output_tensors_train)

            # train the classifier on the training data, test on training and testing data separately
            [model_acc_train, model_acc_test] = classifier.train_and_evaluate_in_phases_train_test(
                output_tensors_reshaped_train,
                ground_truth_ids_train,
                output_tensors_reshaped,
                ground_truth_ids,
                n_classes=self._num_classes,
                device='cuda')

            measurements.add_custom_data(f'weak_class_accuracy_train_{self._layer_id}', model_acc_train)
            measurements.add_custom_data(f'weak_class_accuracy_test_{self._layer_id}', model_acc_test)

        model_acc = classifier.train_and_evaluate_in_phases(output_tensors_reshaped,
                                                            ground_truth_ids,
                                                            n_classes=self._num_classes,
                                                            device='cuda')

        measurements.add_custom_data(f'weak_class_accuracy_{self._layer_id}', model_acc)

        base_acc = classifier.train_and_evaluate_in_phases(random_output_ids,
                                                           ground_truth_ids,
                                                           classifier_input_size=output_size,
                                                           n_classes=self._num_classes,
                                                           device='cuda')

        measurements.add_custom_data(f'base_weak_class_accuracy_{self._layer_id}', base_acc)

        logger.info(f'done')

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


class Task0TaAnalysisComponent(ExperimentComponent):
    _num_classes: int
    _num_layers: int
    _top_layer_id: int

    def __init__(self,
                 topology: Task0TaAnalysisTopology,
                 run_measurement_manager: RunMeasurementManager,
                 experiment_params: Task0TaAnalysisParams,
                 train_test_component: TrainTestMeasuringComponent,
                 num_layers: int,
                 num_classes: int):
        self._run_measurement_manager = run_measurement_manager
        self._train_test_component = train_test_component
        self._params = experiment_params
        self._topology = topology

        self._num_layers = num_layers
        self._num_classes = num_classes
        self._top_layer_id = num_layers - 1

        # value available even during testing
        self._run_measurement_manager.add_measurement_f(
            'dataset_label_ids',
            topology.se_group.get_label_id,
            self._params.measurement_period
        )

        # available during testing
        self._train_test_component.add_measurement_f_testing(
            'dataset_tensors',
            topology.se_group.clone_ground_truth_label_tensor,
            self._params.measurement_period
        )

        # constant zeros, for MSE baseline
        self._train_test_component.add_measurement_f_testing(
            'zeros_baseline_tensor',
            topology.se_group.clone_constant_baseline_output_tensor_for_labels,
            self._params.measurement_period
        )

        # random one-hot vectors
        self._train_test_component.add_measurement_f_testing(
            'random_baseline_tensor',
            topology.se_group.clone_random_baseline_output_tensor_for_labels,
            self._params.measurement_period
        )

        # output of the learned model (TA/NNet)
        self._train_test_component.add_measurement_f_testing(
            'predicted_label_tensor',
            topology.model.clone_predicted_label_tensor_output,
            self._params.measurement_period
        )

        # mostly for debug purposes: learning should be off during testing.
        self._run_measurement_manager.add_measurement_f(
            'is_learning',
            topology.is_learning,
            self._params.measurement_period
        )

        # read output ids for the top-level expert
        output_id_getter = partial(topology.model.get_output_id_for, self._top_layer_id)
        self._train_test_component.add_measurement_f_testing(
            'output_id_' + str(self._top_layer_id),
            output_id_getter,
            self._params.measurement_period
        )

    def calculate_run_results(self):
        super().calculate_run_results()

        measurements = self._run_measurement_manager.measurements
        logger.info('Computing statistics after run...')
        train_test_partitioning = TrainTestMeasurementPartitioning(measurements)

        # ---------- MSE
        baseline_zeros_output = train_test_partitioning.partition_to_list_of_testing_phases('zeros_baseline_tensor')
        model_classification_outputs = train_test_partitioning.partition_to_list_of_testing_phases(
            'predicted_label_tensor')
        dataset_tensors = train_test_partitioning.partition_to_list_of_testing_phases('dataset_tensors')
        baseline_mse = compute_mse_values(dataset_tensors, baseline_zeros_output)
        model_mse = compute_mse_values(dataset_tensors, model_classification_outputs)

        measurements.add_custom_data('baseline_labels_mse', baseline_mse)
        measurements.add_custom_data('predicted_labels_mse', model_mse)

        # ----------- Classification accuracy
        ground_truth_ids = train_test_partitioning.partition_to_list_of_testing_phases('dataset_label_ids')
        random_outputs = train_test_partitioning.partition_to_list_of_testing_phases('random_baseline_tensor')

        base_acc, model_acc = compute_label_reconstruction_accuracies(
            ground_truth_ids=ground_truth_ids,
            baseline_output_tensors=random_outputs,
            model_output_tensors=model_classification_outputs,
            accuracy_method=compute_classification_accuracy,
            num_classes=self._num_classes
        )

        measurements.add_custom_data('baseline_accuracy', base_acc)
        measurements.add_custom_data('model_accuracy', model_acc)

        base_se_acc, model_se_acc = compute_label_reconstruction_accuracies(
            ground_truth_ids=ground_truth_ids,
            baseline_output_tensors=random_outputs,
            model_output_tensors=model_classification_outputs,
            accuracy_method=compute_se_classification_accuracy,
            num_classes=self._num_classes
        )
        measurements.add_custom_data('baseline_se_accuracy', base_se_acc)
        measurements.add_custom_data('model_se_accuracy', model_se_acc)

        # clustering agreement on the top layer
        output_ids = train_test_partitioning.partition_to_list_of_testing_phases(
            'output_id_' + str(self._top_layer_id))

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

        measurements.add_custom_data('base_mutual_info', base_mi)
        measurements.add_custom_data('mutual_info', l1_mi)

        logger.info('run calculations done')


class Task0TaAnalysisTemplate(ExperimentTemplateBase[Task0TaAnalysisTopology]):
    """An experiment which measures classification accuracy on the Task0 and compares with the baseline"""

    def __init__(self,
                 experiment_name: str,
                 experiment_params: Task0TaAnalysisParams,
                 train_test_params: TrainTestComponentParams):
        super().__init__(experiment_name, template=experiment_params, train_test=train_test_params)

        self._experiment_params = experiment_params
        self._train_test_params = train_test_params

    def setup_controller(self,
                         topology: Task0TaAnalysisTopology,
                         controller: ExperimentController,
                         run_measurement_manager: RunMeasurementManager):

        # splits the measurements into train/test phases
        train_test_component = TrainTestControllingComponent(topology,
                                                             run_measurement_manager,
                                                             self._train_test_params)
        controller.register(train_test_component)

        # measures and computes common statistics for the classification
        controller.register(Task0TaAnalysisComponent(topology,
                                                     run_measurement_manager,
                                                     self._experiment_params,
                                                     train_test_component,
                                                     self._experiment_params.num_layers,
                                                     self._experiment_params.num_classes))

        # measures and computes stats for each layer
        for layer_id in range(self._experiment_params.num_layers):
            controller.register(Task0TaAnalysisLayerComponent(topology,
                                                              run_measurement_manager,
                                                              self._experiment_params,
                                                              train_test_component,
                                                              self._experiment_params.num_layers,
                                                              self._experiment_params.num_classes,
                                                              layer_id,
                                                              self._experiment_params.is_train_test_classifier_computed))

    def publish_results(self, document: Document, docs_folder: str, measurement_manager: MeasurementManager,
                        topology_parameters: List[str]):
        """An alternative to the _publish_results method, this is called from _publish_results now

        Draw and add your plots to the document here.
        """
        steps = measurement_manager.get_values_from_all_runs('current_step')
        plotted_training_phase_id = measurement_manager.get_values_from_all_runs('training_phase_id')
        plotted_testing_phase_id = measurement_manager.get_values_from_all_runs('testing_phase_id')
        plotted_is_learning = measurement_manager.get_values_from_all_runs('is_learning')

        labels = topology_parameters
        document.add(f"<br><br><br><b>Common results from the TopExpert</b><br>")

        title = 'training_phase_id'
        plot_multiple_runs(
            steps,
            plotted_training_phase_id,
            title=title,
            ylabel='training_phase_id',
            xlabel='steps',
            labels=labels,
            path=path.join(docs_folder, title),
            doc=document
        )

        title = 'testing_phase_id'
        plot_multiple_runs(
            steps,
            plotted_testing_phase_id,
            title=title,
            ylabel='testing_phase_id',
            xlabel='steps',
            labels=labels,
            path=path.join(docs_folder, title),
            doc=document
        )

        title = 'is_learning'
        plot_multiple_runs(
            steps,
            plotted_is_learning,
            title=title,
            ylabel='is_learning',
            xlabel='steps',
            labels=labels,
            path=path.join(docs_folder, title),
            doc=document
        )

        predicted_labels_mse = measurement_manager.get_custom_data_from_all_runs('predicted_labels_mse')
        testing_phases_x = list(range(0, len(predicted_labels_mse[0])))

        model_accuracy = measurement_manager.get_custom_data_from_all_runs('model_accuracy')
        baseline_accuracy = measurement_manager.get_custom_data_from_all_runs('baseline_accuracy')
        # plot the classification accuracy
        title = 'Label reconstruction accuracy (step-wise)'
        plot_multiple_runs_with_baselines(
            testing_phases_x,
            model_accuracy,
            baseline_accuracy,
            title=title,
            ylabel='accuracy (1 ~ 100%)',
            xlabel='testing phase ID',
            ylim=[-0.1, 1.1],
            labels=labels,
            hide_legend=True,
            path=path.join(docs_folder, title),
            doc=document
        )

        model_se_accuracy = measurement_manager.get_custom_data_from_all_runs('model_se_accuracy')
        baseline_se_accuracy = measurement_manager.get_custom_data_from_all_runs('baseline_se_accuracy')

        # plot the classification accuracy
        title = 'Label reconstruction SE accuracy (object-wise)'
        plot_multiple_runs_with_baselines(
            testing_phases_x,
            model_se_accuracy,
            baseline_se_accuracy,
            title=title,
            ylabel='accuracy (1 ~ 100%)',
            xlabel='testing_phase ID',
            ylim=[-0.1, 1.1],
            labels=labels,
            hide_legend=True,
            path=path.join(docs_folder, title),
            doc=document
        )

        baseline_labels_mse = measurement_manager.get_custom_data_from_all_runs('baseline_labels_mse')

        # plot the MSE
        title = 'Mean Square Error of label reconstruction'
        plot_multiple_runs_with_baselines(
            testing_phases_x,
            predicted_labels_mse,
            baseline_labels_mse,
            title=title,
            ylabel='MSE',
            xlabel='testing phase ID',
            labels=labels,
            hide_legend=True,
            ylim=[-0.1, 0.2],  # just for better resolution
            path=path.join(docs_folder, title),
            doc=document
        )

        for layer_id in reversed(range(self._experiment_params.num_layers)):
            self.publish_results_for_layer(document,
                                           docs_folder,
                                           measurement_manager,
                                           topology_parameters,
                                           layer_id,
                                           self._experiment_params.num_layers,
                                           self._experiment_params.show_conv_agreements,
                                           self._experiment_params.is_train_test_classifier_computed)

    @staticmethod
    def publish_results_for_layer(document: Document,
                                  docs_folder: str,
                                  measurement_manager: MeasurementManager,
                                  topology_parameters: List[str],
                                  layer_id: int,
                                  num_layers: int,
                                  show_conv_agreements: bool,
                                  is_train_test_classifier_computed: bool):
        """Publish results for each layer separately

        This uses the data measured and computed for each run by the Task0TaAnalysisLayerComponent and aggregated
        and stored in the measurement_manager.
        """

        logger.info(f'publishing results for layer {layer_id}...')

        num_boosted_clusters = measurement_manager.get_values_from_all_runs(f'num_boosted_getter_{layer_id}')
        average_boosting_dur = measurement_manager.get_values_from_all_runs(f'average_boosting_dur_{layer_id}')
        average_deltas = measurement_manager.get_values_from_all_runs(f'average_delta_{layer_id}')

        base_weak_class_accuracy = measurement_manager.get_custom_data_from_all_runs(
            f'base_weak_class_accuracy_{layer_id}')
        clustering_agreements = measurement_manager.get_custom_data_from_all_runs(f'clustering_agreements_{layer_id}')

        average_steps_deltas = measurement_manager.get_items_from_all_runs(f'average_delta_{layer_id}')
        sp_evaluation_steps = [key for key, value in average_steps_deltas[0]]

        labels = topology_parameters

        document.add(f"<br><br><br><b>Results for layer {layer_id}</b><br>")
        prefix = 'L' + str(layer_id) + '--'

        testing_phase_ids = list(range(0, len(clustering_agreements[0][0])))  # x-axis values

        if is_train_test_classifier_computed:
            weak_class_accuracy_train = measurement_manager.get_custom_data_from_all_runs(
                f'weak_class_accuracy_train_{layer_id}')
            weak_class_accuracy_test = measurement_manager.get_custom_data_from_all_runs(
                f'weak_class_accuracy_test_{layer_id}')

            title = prefix + ' Weak classifier accuracy (trained on train, tested on train data)'
            plot_multiple_runs_with_baselines(
                testing_phase_ids,
                weak_class_accuracy_train,
                base_weak_class_accuracy,
                title=title,
                ylabel='Accuracy (1 ~ 100%)',
                xlabel='steps',
                labels=labels,
                ylim=[-0.1, 1.1],
                hide_legend=True,
                path=path.join(docs_folder, title),
                doc=document
            )

            title = prefix + ' Weak classifier accuracy (trained on train, tested on test data)'
            plot_multiple_runs_with_baselines(
                testing_phase_ids,
                weak_class_accuracy_test,
                base_weak_class_accuracy,
                title=title,
                ylabel='Accuracy (1 ~ 100%)',
                xlabel='steps',
                labels=labels,
                ylim=[-0.1, 1.1],
                hide_legend=True,
                path=path.join(docs_folder, title),
                doc=document
            )

        weak_class_accuracy = measurement_manager.get_custom_data_from_all_runs(f'weak_class_accuracy_{layer_id}')
        title = prefix + ' Weak classifier accuracy (trained on test, tested on test)'
        plot_multiple_runs_with_baselines(
            testing_phase_ids,
            weak_class_accuracy,
            base_weak_class_accuracy,
            title=title,
            ylabel='Accuracy (1 ~ 100%)',
            xlabel='steps',
            labels=labels,
            ylim=[-0.1, 1.1],
            hide_legend=True,
            path=path.join(docs_folder, title),
            doc=document
        )

        title = prefix + '- Average boosting duration'
        plot_multiple_runs(
            sp_evaluation_steps,
            average_boosting_dur,
            title=title,
            ylabel='duration',
            xlabel='steps',
            labels=labels,
            hide_legend=True,
            path=path.join(docs_folder, title),
            doc=document
        )

        title = prefix + '- Num boosted clusters'
        plot_multiple_runs(
            sp_evaluation_steps,
            num_boosted_clusters,
            title=title,
            ylabel='Num boosted clusters / total clusters',
            xlabel='steps',
            labels=labels,
            ylim=[-0.1, 1.1],
            hide_legend=True,
            path=path.join(docs_folder, title),
            doc=document

        )

        title = prefix + '- Average_deltas'
        plot_multiple_runs(
            sp_evaluation_steps,
            average_deltas,
            title=title,
            ylabel='average_deltas',
            xlabel='steps',
            labels=labels,
            disable_ascii_labels=True,
            hide_legend=True,
            # use_scatter=True,
            path=path.join(docs_folder, title),
            doc=document
        )

        # if this is not the top layer, show conv agreements only if required
        if show_conv_agreements or layer_id == (num_layers - 1):
            agreements = clustering_agreements

            for run_id, run_agreements in enumerate(agreements):
                for expert_id, expert_agreements in enumerate(run_agreements):
                    Task0TaAnalysisTemplate._plot_agreement(
                        prefix,
                        expert_id,
                        run_id,
                        testing_phase_ids,
                        expert_agreements,
                        document,
                        docs_folder
                    )

        logger.info('done')

    @staticmethod
    def _plot_agreement(prefix: str,
                        expert_id: int,
                        run_id: int,
                        testing_phase_ids: List[int],
                        agreements: List[List[int]],
                        doc,
                        docs_folder):
        """Plot cluster agreements for one expert in a layer"""

        title = prefix + f'- E{expert_id}-Run{run_id} - Clustering agreements'
        plot_multiple_runs(
            testing_phase_ids,
            agreements,
            title=title,
            ylabel='agreement',
            xlabel=f'testing phases',
            disable_ascii_labels=True,
            hide_legend=True,
            ylim=[ClusterAgreementMeasurement.NO_VALUE - 0.1, 1.1],
            path=path.join(docs_folder, title),
            doc=doc
        )
