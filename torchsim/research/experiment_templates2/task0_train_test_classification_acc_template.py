import logging
from functools import partial
from os import path
from typing import NamedTuple, List

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval2.experiment_controller import ExperimentController, TrainTestComponentParams, ExperimentComponent, \
    TrainTestMeasuringComponent, TrainTestControllingComponent
from torchsim.core.eval2.experiment_template_base import ExperimentTemplateBase
from torchsim.core.eval2.measurement_manager import RunMeasurementManager, MeasurementManager
from torchsim.core.eval2.run_measurement import TrainTestMeasurementPartitioning
from torchsim.core.eval.series_plotter import plot_multiple_runs, add_fig_to_doc, plot_multiple_runs_with_baselines
from torchsim.research.research_topics.rt_2_1_2_learning_rate2.topologies.classification_accuracy_modular_topology import \
    ClassificationAccuracyModularTopology
from torchsim.utils.template_utils.template_helpers import compute_classification_accuracy, \
    compute_se_classification_accuracy, compute_label_reconstruction_accuracies, compute_mse_values

logger = logging.getLogger(__name__)


class Task0TrainTestClassificationAccParams(NamedTuple):
    measurement_period: int
    num_classes: int
    sp_evaluation_period: int


class Task0TrainTestClassificationAccComponent(ExperimentComponent):
    def __init__(self,
                 topology: ClassificationAccuracyModularTopology,
                 run_measurement_manager: RunMeasurementManager,
                 experiment_params: Task0TrainTestClassificationAccParams,
                 train_test_component: TrainTestMeasuringComponent):

        self._run_measurement_manager = run_measurement_manager
        self._params = experiment_params

        self._train_test_component = train_test_component

        # value available even during testing
        self._train_test_component.add_measurement_f_testing(
            'dataset_label_ids',
            topology.get_label_id,
            self._params.measurement_period
        )

        # available during testing
        self._train_test_component.add_measurement_f_testing(
            'dataset_tensors',
            topology.clone_ground_truth_label_tensor,
            self._params.measurement_period
        )

        # constant zeros, for MSE baseline
        self._train_test_component.add_measurement_f_testing(
            'zeros_baseline_tensor',
            topology.clone_constant_baseline_output_tensor_for_labels,
            self._params.measurement_period
        )

        # random one-hot vectors
        self._train_test_component.add_measurement_f_testing(
            'random_baseline_tensor',
            topology.clone_random_baseline_output_tensor_for_labels,
            self._params.measurement_period
        )

        # output of the learned model (TA/NNet)
        self._train_test_component.add_measurement_f_testing(
            'predicted_label_tensor',
            topology.clone_predicted_label_tensor_output,
            self._params.measurement_period
        )

        # average deltas (just to visually check that the model converges?)
        delta_getter = partial(topology.get_average_log_delta_for, 0)
        self._train_test_component.add_measurement_f_testing(
            'average_delta0_train',
            delta_getter,
            self._params.sp_evaluation_period
        )
        delta_getter_1 = partial(topology.get_average_log_delta_for, 1)
        self._train_test_component.add_measurement_f_testing(
            'average_delta1_train',
            delta_getter_1,
            self._params.sp_evaluation_period
        )

    def calculate_run_results(self):
        super().calculate_run_results()

        measurements = self._run_measurement_manager.measurements

        logger.info('computing statistics after run...')

        # ---------- MSE
        train_test_partitioning = TrainTestMeasurementPartitioning(measurements)

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
            num_classes=self._params.num_classes
        )
        measurements.add_custom_data('baseline_accuracy', base_acc)
        measurements.add_custom_data('model_accuracy', model_acc)

        base_se_acc, model_se_acc = compute_label_reconstruction_accuracies(
            ground_truth_ids=ground_truth_ids,
            baseline_output_tensors=random_outputs,
            model_output_tensors=model_classification_outputs,
            accuracy_method=compute_se_classification_accuracy,
            num_classes=self._params.num_classes
        )
        measurements.add_custom_data('baseline_se_accuracy', base_se_acc)
        measurements.add_custom_data('model_se_accuracy', model_se_acc)

        logger.info('run calculations done')


class Task0TrainTestClassificationAccTemplate(ExperimentTemplateBase[ClassificationAccuracyModularTopology]):
    """An experiment which measures classification accuracy on the Task0 and compares with the baseline"""

    def __init__(self,
                 experiment_name: str,
                 experiment_params: Task0TrainTestClassificationAccParams,
                 train_test_params: TrainTestComponentParams):
        super().__init__(experiment_name, template=experiment_params, train_test=train_test_params)

        self._experiment_params = experiment_params
        self._train_test_params = train_test_params

    def setup_controller(self, topology: ClassificationAccuracyModularTopology, controller: ExperimentController,
                         run_measurement_manager: RunMeasurementManager):
        train_test_component = TrainTestControllingComponent(topology, run_measurement_manager, self._train_test_params)

        controller.register(train_test_component)
        controller.register(Task0TrainTestClassificationAccComponent(topology,
                                                                     run_measurement_manager,
                                                                     self._experiment_params,
                                                                     train_test_component))

    def publish_results(self, document: Document, docs_folder: str, measurement_manager: MeasurementManager,
                        topology_parameters: List[str]):
        """An alternative to the _publish_results method, this is called from _publish_results now

        Draw and add your plots to the document here.
        """
        steps = measurement_manager.get_values_from_all_runs('current_step')
        plotted_training_phase_id = measurement_manager.get_values_from_all_runs('training_phase_id')
        plotted_testing_phase_id = measurement_manager.get_values_from_all_runs('testing_phase_id')

        labels = topology_parameters

        title = 'training_phase_id'
        f = plot_multiple_runs(
            steps,
            plotted_training_phase_id,
            title=title,
            ylabel='training_phase_id',
            xlabel='steps',
            labels=labels
        )
        add_fig_to_doc(f, path.join(docs_folder, title), document)

        title = 'testing_phase_id'
        f = plot_multiple_runs(
            steps,
            plotted_testing_phase_id,
            title=title,
            ylabel='testing_phase_id',
            xlabel='steps',
            labels=labels
        )
        add_fig_to_doc(f, path.join(docs_folder, title), document)

        max_x = max(steps[0])
        average_delta_train_0 = measurement_manager.get_items_from_all_runs('average_delta0_train')
        test_sp_steps = [key for key, value in average_delta_train_0[0]]

        title = 'average_delta_train_layer0'
        f = plot_multiple_runs(
            test_sp_steps,
            [[value for key, value in sequence] for sequence in average_delta_train_0],
            title=title,
            ylabel='average_deltas',
            xlabel='steps',
            labels=labels,
            xlim=[0, max_x],
            disable_ascii_labels=True,
            hide_legend=True
            # use_scatter=True
        )
        add_fig_to_doc(f, path.join(docs_folder, title), document)

        average_delta_train_1 = measurement_manager.get_items_from_all_runs('average_delta1_train')
        title = 'average_delta_train_layer1'
        f = plot_multiple_runs(
            test_sp_steps,
            [[value for key, value in sequence] for sequence in average_delta_train_1],
            title=title,
            ylabel='average_deltas',
            xlabel='steps',
            labels=labels,
            xlim=[0, max_x],
            disable_ascii_labels=True,
            hide_legend=True
            # use_scatter=True
        )
        add_fig_to_doc(f, path.join(docs_folder, title), document)

        predicted_labels_mse = measurement_manager.get_custom_data_from_all_runs('predicted_labels_mse')
        testing_phases_x = list(range(0, len(predicted_labels_mse[0])))

        model_accuracy = measurement_manager.get_custom_data_from_all_runs('model_accuracy')
        baseline_accuracy = measurement_manager.get_custom_data_from_all_runs('baseline_accuracy')
        # plot the classification accuracy
        title = 'Label reconstruction accuracy (step-wise)'
        f = plot_multiple_runs_with_baselines(
            testing_phases_x,
            model_accuracy,
            baseline_accuracy,
            title=title,
            ylabel='accuracy (1 ~ 100%)',
            xlabel='testing phase ID',
            ylim=[-0.1, 1.1],
            labels=labels,
            hide_legend=True
        )
        add_fig_to_doc(f, path.join(docs_folder, title), document)

        model_se_accuracy = measurement_manager.get_custom_data_from_all_runs('model_se_accuracy')
        baseline_se_accuracy = measurement_manager.get_custom_data_from_all_runs('baseline_se_accuracy')

        # plot the classification accuracy
        title = 'Label reconstruction SE accuracy (step-wise)'
        f = plot_multiple_runs_with_baselines(
            testing_phases_x,
            model_se_accuracy,
            baseline_se_accuracy,
            title=title,
            ylabel='accuracy (1 ~ 100%)',
            xlabel='testing_phase ID',
            ylim=[-0.1, 1.1],
            labels=labels,
            hide_legend=True
        )
        add_fig_to_doc(f, path.join(docs_folder, title), document)

        baseline_labels_mse = measurement_manager.get_custom_data_from_all_runs('baseline_labels_mse')

        # plot the MSE
        title = 'Mean Square Error of label reconstruction'
        f = plot_multiple_runs_with_baselines(
            testing_phases_x,
            predicted_labels_mse,
            baseline_labels_mse,
            title=title,
            ylabel='MSE',
            xlabel='testing phase ID',
            labels=labels,
            hide_legend=True,
            ylim=[-0.1, 0.2]  # just for better resolution
        )
        add_fig_to_doc(f, path.join(docs_folder, title), document)
