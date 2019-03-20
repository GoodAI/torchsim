import logging
import pprint
from abc import abstractmethod
from os import path
from typing import List, Tuple, Union, Dict, Any

import torch
from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval.experiment_template_base import ExperimentTemplateBase
from torchsim.core.eval.metrics.simple_classifier_svm import SvmClassifier
from torchsim.core.eval.series_plotter import add_fig_to_doc, plot_multiple_runs, \
    plot_multiple_runs_with_baselines
from torchsim.research.experiment_templates.task0_train_test_template_base import Task0TrainTestTemplateBase, \
    Task0TrainTestTemplateAdapterBase
from torchsim.research.research_topics.rt_2_1_2_learning_rate.utils.cluster_agreement_measurement import \
    ClusterAgreementMeasurement
from torchsim.utils.template_utils.template_helpers import compute_se_classification_accuracy, \
    compute_classification_accuracy, argmax_tensor, argmax_list_list_tensors

logger = logging.getLogger(__name__)


class AbstractRelearnAdapter(Task0TrainTestTemplateAdapterBase):
    @abstractmethod
    def get_sp_output_id(self) -> int:
        pass

    @abstractmethod
    def get_label_id(self) -> int:
        pass

    @abstractmethod
    def get_sp_size(self) -> int:
        pass

    @abstractmethod
    def clone_random_baseline_output_tensor_for_labels(self) -> torch.Tensor:
        pass

    @abstractmethod
    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_average_log_delta(self) -> float:
        pass

    @abstractmethod
    def get_average_boosting_duration(self) -> float:
        pass


class Task0TrainTestTemplateRelearning(Task0TrainTestTemplateBase):
    _topology_adapter: AbstractRelearnAdapter

    def __init__(self,
                 topology_adapter: AbstractRelearnAdapter,
                 topology_class,
                 topology_parameters: Union[List[Tuple], List[Dict]],
                 run_labels: List[str],
                 overall_training_steps: int,
                 num_testing_steps: int,
                 num_testing_phases: int,
                 measurement_period: int = 5,
                 sp_evaluation_period: int = 10,
                 seed=None,
                 experiment_name: str = 'empty_name',
                 learning_rate: float = 0.01,
                 save_cache=True,
                 load_cache=True,
                 clear_cache=True,
                 computation_only=False,
                 results_folder=None,
                 disable_plt_show=True):
        super().__init__(topology_adapter=topology_adapter,
                         topology_class=topology_class,
                         sp_evaluation_period=sp_evaluation_period,
                         measurement_period=measurement_period,
                         topology_parameters=topology_parameters,
                         overall_training_steps=overall_training_steps,
                         num_testing_steps=num_testing_steps,
                         num_testing_phases=num_testing_phases,
                         save_cache=save_cache,
                         load_cache=load_cache,
                         computation_only=computation_only,
                         seed=seed,
                         disable_plt_show=disable_plt_show,
                         experiment_folder=results_folder,
                         clear_cache=clear_cache)
        self._experiment_name = experiment_name
        self._measurement_manager = self._create_measurement_manager(self._experiment_folder,
                                                                     delete_after_each_run=False)

        self._clustering_agreement_list = []
        self._run_labels = run_labels
        self._learning_rate = learning_rate

        self._measurement_manager.add_measurement_f_with_period_testing(
            'sp_output_ids',
            self._topology_adapter.get_sp_output_id,
            self._measurement_period
        )

        # value available even during testing
        self._measurement_manager.add_measurement_f_with_period_testing(
            'dataset_label_ids',
            topology_adapter.get_label_id,
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

        self._measurement_manager.add_measurement_f_with_period_training(
            'average_delta0_train',
            topology_adapter.get_average_log_delta,
            self._sp_evaluation_period
        )

        self._measurement_manager.add_measurement_f_with_period_testing(
            'average_delta0_test',
            topology_adapter.get_average_log_delta,
            self._sp_evaluation_period
        )

        self._measurement_manager.add_measurement_f_with_period_training(
            'average_boosting_dur',
            self._topology_adapter.get_average_boosting_duration,
            self._sp_evaluation_period
        )

        self._model_accuracy = []
        self._baseline_accuracy = []
        self._model_se_accuracy = []
        self._baseline_se_accuracy = []
        self._average_delta_train = []
        self._average_delta_test = []

        self._weak_class_accuracy = []
        self._base_weak_class_accuracy = []
        self._average_boosting_dur = []

        # only this is collected during both training and testing
        # self._measurement_manager.add_measurement_f_with_period(
        #     'is_learning',
        #     self._topology_adapter.is_learning,
        #     self._measurement_period
        # )

    def experiment_name(self):
        return self._experiment_name

    def _complete_name(self):
        return f"{self._topology_class.__name__}_" + self._experiment_name + "_"

    def _after_run_finished(self):
        last_run_measurements = self._measurement_manager.run_measurements[-1]

        self._average_delta_test.append([])
        self._average_delta_train.append([])
        self._average_boosting_dur.append([])

        # go step-by-step through the last run (single_measurement contains all the values taken in that time-step)
        for single_measurement in last_run_measurements:
            if 'average_delta0_train' in single_measurement:
                self._average_delta_train[-1].append(single_measurement['average_delta0_train'])
            if 'average_delta0_test' in single_measurement:
                self._average_delta_test[-1].append(single_measurement['average_delta0_test'])
            if 'average_boosting_dur' in single_measurement:
                self._average_boosting_dur[-1].append(single_measurement['average_boosting_dur'])

        output_ids = last_run_measurements.partition_to_list_of_testing_phases('sp_output_ids')

        agreements = ClusterAgreementMeasurement.compute_cluster_agreements(Task0TrainTestTemplateRelearning.
                                                                            _to_long_tensors(output_ids))
        self._clustering_agreement_list.append(agreements)

        model_classification_outputs = last_run_measurements.partition_to_list_of_testing_phases(
            'predicted_label_tensor'
        )

        # ----------- Classification accuracy
        ground_truth_ids = last_run_measurements.partition_to_list_of_testing_phases('dataset_label_ids')
        random_outputs = last_run_measurements.partition_to_list_of_testing_phases('random_baseline_tensor')
        base_acc, model_acc = self._compute_accuracies(
            ground_truth_ids=ground_truth_ids,
            baseline_output_tensors=random_outputs,
            model_output_tensors=model_classification_outputs,
            accuracy_method=compute_classification_accuracy
        )
        self._baseline_accuracy.append(base_acc)
        self._model_accuracy.append(model_acc)
        base_se_acc, model_se_acc = self._compute_accuracies(
            ground_truth_ids=ground_truth_ids,
            baseline_output_tensors=random_outputs,
            model_output_tensors=model_classification_outputs,
            accuracy_method=compute_se_classification_accuracy
        )
        self._baseline_se_accuracy.append(base_se_acc)
        self._model_se_accuracy.append(model_se_acc)

        random_outputs = last_run_measurements.partition_to_list_of_testing_phases('random_baseline_tensor')
        random_output_ids = argmax_list_list_tensors(random_outputs)

        # weak classifier accuracy
        n_classes = 20  # hard code it for now
        output_size = self._topology_adapter.get_sp_size()

        model_acc = SvmClassifier().train_and_evaluate_in_phases(output_ids,
                                                                 ground_truth_ids,
                                                                 classifier_input_size=output_size,
                                                                 n_classes=n_classes,
                                                                 device='cuda')
        self._weak_class_accuracy.append(model_acc)

        base_acc = SvmClassifier().train_and_evaluate_in_phases(random_output_ids,
                                                                ground_truth_ids,
                                                                classifier_input_size=output_size,
                                                                n_classes=n_classes,
                                                                device='cuda')
        self._base_weak_class_accuracy.append(base_acc)

        print('done')

    def _compute_accuracies(self,
                            ground_truth_ids: List[List[int]],
                            baseline_output_tensors: List[List[torch.Tensor]],
                            model_output_tensors: List[List[torch.Tensor]],
                            accuracy_method) -> (List[List[float]], List[List[float]]):
        base_acc = self._compute_classification_accuracy(
            label_ids=ground_truth_ids,
            output_tensors=baseline_output_tensors,
            accuracy_method=accuracy_method,
            num_classes=20
        )
        model_acc = self._compute_classification_accuracy(
            label_ids=ground_truth_ids,
            output_tensors=model_output_tensors,
            accuracy_method=accuracy_method,
            num_classes=20
        )
        return base_acc, model_acc

    @staticmethod
    def _compute_classification_accuracy(label_ids: List[List[int]],
                                         output_tensors: List[List[torch.Tensor]],
                                         accuracy_method,
                                         num_classes: int) -> List[List[float]]:
        """ Compute classification accuracy for collected outputs
        Args:
            label_ids: List (for testing phase) of Lists of ints (id ~ each measurement in the phase)
            output_tensors: List (for testing phase) of Lists of tensors (tensor ~ output measured)
         Returns:
             List (for testing phase) of floats (accuracy)
        """
        output_ids = Task0TrainTestTemplateRelearning.argmax_list_list_tensors(output_tensors)
        assert len(output_ids) == len(label_ids)
        phase_accuracies = []
        for phase_label_ids, phase_output_ids in zip(label_ids, output_ids):
            phase_accuracies.append(accuracy_method(phase_label_ids, phase_output_ids, num_classes=num_classes))
        return phase_accuracies

    @staticmethod
    def argmax_list_list_tensors(tensors: List[List[torch.Tensor]]) -> List[List[int]]:
        """
        Convert list of tensors to list of their argmaxes
        Args:
            tensors: List[List[tensor]] outputs of the classifier for each phase
        Returns: list of lists of output ids (argmax)
        """
        result = []
        for phase in tensors:
            ids_in_phase = []
            for tensor in phase:
                ids_in_phase.append(argmax_tensor(tensor))
            result.append(ids_in_phase)
        return result

    @staticmethod
    def _to_long_tensors(model_outputs: List[List[int]]) -> List[torch.Tensor]:
        result = []

        for single_phase_outputs in model_outputs:
            result.append(torch.tensor(single_phase_outputs, dtype=torch.long))

        return result

    def _compute_experiment_statistics(self):
        logger.error("not implemented")

    @staticmethod
    def pprint_parameters_to_string(parameters: List[Dict[str, Any]]) -> List[str]:
        return [", ".join(
            f"{param}: {pprint.pformat(value, 1, 160)}" for param, value in parameter.items()
            if "training_phase_steps" not in param and "testing_phase_steps" not in param
        ) for parameter in parameters]

    def _publish_results(self):
        doc = Document()
        doc.add(self._get_heading('_not set'))

        params_str = pprint.pformat(
            Task0TrainTestTemplateRelearning.pprint_parameters_to_string(self._topology_parameters_list), 1, 160
        )

        doc.add(f"<p><b>Parameters:</b><br /><pre>learning rate: {self._learning_rate},\n" + params_str + "</pre></p>")

        labels = ExperimentTemplateBase.parameters_to_string(self._topology_parameters_list)

        testing_phases = list(range(0, len(self._clustering_agreement_list[0][0])))

        for i, run_label in enumerate(self._run_labels):
            title = 'Clustering agreement ' + run_label
            f = plot_multiple_runs(
                testing_phases,
                self._clustering_agreement_list[i],
                title=title,
                ylabel='agreement',
                xlabel='testing training_phases',
                disable_ascii_labels=True,
                hide_legend=True,
                ylim=[ClusterAgreementMeasurement.NO_VALUE - 0.1, 1.1]
            )
            add_fig_to_doc(f, path.join(self._docs_folder, title), doc)

        title = 'Model classification accuracy (step-wise)'
        f = plot_multiple_runs_with_baselines(
            testing_phases,
            self._model_accuracy,
            self._baseline_accuracy,
            title=title,
            ylabel='accuracy (1 ~ 100%)',
            xlabel='testing phase',
            ylim=[-0.1, 1.1],
            labels=self._run_labels
        )
        add_fig_to_doc(f, path.join(self._docs_folder, title), doc)
        title = 'Model SE classification accuracy (step-wise)'
        f = plot_multiple_runs_with_baselines(
            testing_phases,
            self._model_se_accuracy,
            self._baseline_se_accuracy,
            title=title,
            ylabel='accuracy (1 ~ 100%)',
            xlabel='testing_phase',
            ylim=[-0.1, 1.1],
            labels=self._run_labels
        )
        add_fig_to_doc(f, path.join(self._docs_folder, title), doc)

        title = 'Weak classifier accuracy trained on SP outputs to labels'
        f = plot_multiple_runs_with_baselines(
            testing_phases,
            self._weak_class_accuracy,
            self._base_weak_class_accuracy,
            title=title,
            ylabel='Accuracy (1 ~ 100%)',
            xlabel='steps',
            labels=labels,
            ylim=[-0.1, 1.1],
            hide_legend=True
        )
        add_fig_to_doc(f, path.join(self._docs_folder, title), doc)

        title = 'Average Deltas Train'
        f = plot_multiple_runs(
            [self._sp_evaluation_period * i for i in range(0, len(self._average_delta_train[0]))],
            self._average_delta_train,
            title=title,
            ylabel='average_deltas',
            xlabel='steps',
            labels=self._run_labels,
            disable_ascii_labels=True,
            use_scatter=False
        )
        add_fig_to_doc(f, path.join(self._docs_folder, title), doc)

        title = 'Average Deltas Test'
        f = plot_multiple_runs(
            [self._sp_evaluation_period * i for i in range(0, len(self._average_delta_test[0]))],
            self._average_delta_test,
            title=title,
            ylabel='average_deltas',
            xlabel='steps',
            labels=self._run_labels,
            disable_ascii_labels=True,
            use_scatter=False
        )
        add_fig_to_doc(f, path.join(self._docs_folder, title), doc)

        title = 'Average boosting duration'
        f = plot_multiple_runs(
            [self._sp_evaluation_period * i for i in range(0, len(self._average_boosting_dur[0]))],
            self._average_boosting_dur,
            title=title,
            ylabel='duration',
            xlabel='steps',
            labels=labels,
            hide_legend=True
        )
        add_fig_to_doc(f, path.join(self._docs_folder, title), doc)

        doc.write_file(path.join(self._docs_folder, "main.html"))
        print('done')
