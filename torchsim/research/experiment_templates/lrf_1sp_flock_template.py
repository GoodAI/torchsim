from os import path

import numpy as np
import torch
from abc import abstractmethod

from eval_utils import progress_bar
from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval.experiment_template_base import ExperimentTemplateBase
from torchsim.core.eval.metrics.entropy import one_hot_entropy
from torchsim.core.eval.metrics.mean_squared_error import mse_loss
from torchsim.core.eval.metrics.simple_classifier_svm import compute_svm_classifier_accuracy
from torchsim.core.eval.series_plotter import plot_multiple_runs, get_stamp, to_safe_name, add_fig_to_doc
from torchsim.core.eval.topology_adapter_base import TopologyAdapterBase


class Lrf1SpFlockTemplate(TopologyAdapterBase):
    """A general subject for the experiment: LRF_1SPFlockExperiment."""

    @abstractmethod
    def get_reconstructed_image(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_input_image(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_difference_image(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_memory_used(self) -> float:
        pass

    @abstractmethod
    def get_label(self) -> float:
        pass

    @abstractmethod
    def get_sp_output(self) -> float:
        pass

    @abstractmethod
    def get_is_testing(self) -> float:
        pass

    @abstractmethod
    def get_testing_phase_number(self) -> int:
        pass

    @abstractmethod
    def get_training_step(self) -> int:
        pass


class Lrf1SpFlockExperimentTemplate(ExperimentTemplateBase):

    _measurement_period: int

    _evaluation_period: int
    _steps: np.array

    _topology_adapter: Lrf1SpFlockTemplate  # the subject of the experiment

    _seed: int

    def __init__(self,
                 topology_adapter: Lrf1SpFlockTemplate,
                 topology_class,
                 models_params,
                 max_steps: int,
                 measurement_period: int,
                 save_cache=True,
                 load_cache=True,
                 clear_cache=True,
                 computation_only=False,
                 experiment_folder=None):
        super().__init__(
            topology_adapter,
            topology_class,
            models_params,
            max_steps=max_steps,
            save_cache=save_cache,
            load_cache=load_cache,
            computation_only=computation_only,
            experiment_folder=experiment_folder,
            clear_cache=clear_cache
        )

        self._measurement_manager = self._create_measurement_manager(self._experiment_folder, delete_after_each_run=True,
                                                                     zip_data=True)
        self._measurement_manager.add_measurement_f_with_period(
            'input_image',
            topology_adapter.get_input_image,
            measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'reconstructed_image',
            topology_adapter.get_reconstructed_image,
            measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'label',
            topology_adapter.get_label,
            measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'testing_phase',
            topology_adapter.get_is_testing,
            measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'testing_phase_number',
            topology_adapter.get_testing_phase_number,
            measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'training_step',
            topology_adapter.get_training_step,
            measurement_period
        )
        self._measurement_manager.add_measurement_f_with_period(
            'forward_clusters',
            topology_adapter.get_sp_output,
            measurement_period
        )
        self._measurement_manager.add_measurement_f_once(
            'memory_used',
            topology_adapter.get_memory_used,
            0
        )

        self._classifier_window_length = min(max_steps / 10, 200)

        self._mse = []
        self._mse_testing = []
        self._weak_classifier_results = []
        self._training_steps_before_testing = []
        self._memory_used = []
        self._code_entropy = []

    def _get_measurement_manager(self):
        return self._measurement_manager

    def _after_run_finished(self):
        print(f'--- computing statistics after run')
        self._mse.append([])
        self._mse_testing.append([])
        self._weak_classifier_results.append([])
        self._training_steps_before_testing.append([])
        self._code_entropy.append([])
        run_measurement = self._measurement_manager.run_measurements[-1]
        testing_phases = []

        for measurement in run_measurement:
            # if switch from training to testing is detected, start a new testing_phase entry
            if measurement['testing_phase_number'] + 1 > len(testing_phases) and measurement['testing_phase']:
                testing_phases.append([])
                self._training_steps_before_testing[-1].append(measurement['training_step'])

            if not measurement['testing_phase']:
                self._mse[-1].append(
                    mse_loss(
                        measurement['input_image'],
                        measurement['reconstructed_image']
                    ).item()
                )

            if measurement['testing_phase']:
                testing_phases[-1].append((measurement['forward_clusters'], measurement['label'],
                                          measurement['input_image'], measurement['reconstructed_image']))

        for testing_phase in progress_bar(testing_phases):
            clusters, labels, images, reconstructed_images = zip(*testing_phase)
            clusters = torch.stack(clusters)
            #
            if not isinstance(labels[0], torch.Tensor):
                labels = torch.tensor(labels)
            else:
                labels = torch.stack(labels)
            self._weak_classifier_results[-1].append(
                compute_svm_classifier_accuracy(
                    clusters.view(clusters.shape[0], -1),
                    labels,
                    n_classes=10)
            )

            self._mse_testing[-1].append(
                mse_loss(
                    torch.stack(images),
                    torch.stack(reconstructed_images)
                ).item()
            )

            self._code_entropy[-1].append(one_hot_entropy(clusters))

        self._memory_used.append(run_measurement.get_item('memory_used', 0))

    # @staticmethod
    # def _move_tensors_to_cpu(list_of_tensors):
    #     """Recursively."""
    #     if isinstance(list_of_tensors, torch.Tensor):
    #         return list_of_tensors.cpu()
    #
    #     return list(map(lambda x: LRF_1SPFlockExperimentTemplate._move_list_to_cpu(x), list_of_tensors))

    def _compute_experiment_statistics(self):
        """Compute statistics from the collected measurements."""

        # self._mse = self._move_list_to_cpu(self._mse)
        # self._mse_testing = self._move_list_to_cpu(self._mse_testing)
        # self._weak_classifier_results = self._move_list_to_cpu(self._weak_classifier_results)
        # self._training_steps_before_testing = self._move_list_to_cpu(self._training_steps_before_testing)
        # self._memory_used = self._move_list_to_cpu(self._memory_used)

    def _experiment_template_name(self):
        return self._topology_adapter.__class__.__name__

    def _publish_results(self):
        """Plot and optionally save the results."""

        mse = np.array(self._mse)
        mse_testing = np.array(self._mse_testing)

        memory_used = torch.tensor(self._memory_used, dtype=torch.float32)
        window_size = 201
        memory_used = (memory_used.view(-1, 1).expand(mse_testing.shape) / (1024 ** 2))
        error_memory_ratio = torch.tensor(mse_testing, dtype=torch.float32) * memory_used
        accuracy_memory_ratio = []
        for run_acc, run_mem in zip(self._weak_classifier_results, memory_used):
            accuracy_memory_ratio_run = []
            for acc, mem in zip(run_acc, run_mem):
                accuracy_memory_ratio_run.append((1 - acc) * mem)

            accuracy_memory_ratio.append(accuracy_memory_ratio_run)

        accuracy_memory_ratio = torch.tensor(accuracy_memory_ratio)

        doc = Document()

        figs = []
        xlabel = "steps"
        ylabel = "mean reconstruction error"
        title = "Influence of hyperparameters on reconstruction error (training)"
        figsize = (18, 12)
        date = get_stamp()

        fig = plot_multiple_runs(x_values=np.arange(len(mse[0])),
                                 y_values=mse,
                                 labels=ExperimentTemplateBase.parameters_to_string(self._topology_parameters_list),
                                 smoothing_window_size=window_size,
                                 xlabel=xlabel,
                                 ylabel=ylabel,
                                 title=title,
                                 figsize=figsize,
                                 hide_legend=True
                                 )
        add_fig_to_doc(fig, path.join(self._docs_folder, to_safe_name(title)), doc)

        fig = plot_multiple_runs(x_values=np.arange(len(mse[0])),
                                 y_values=mse,
                                 labels=ExperimentTemplateBase.parameters_to_string(self._topology_parameters_list),
                                 xlabel=xlabel,
                                 ylabel=ylabel,
                                 title=title,
                                 figsize=figsize,
                                 hide_legend=True
                                 )
        add_fig_to_doc(fig, path.join(self._docs_folder, to_safe_name(title) + "_smooth"), doc)

        title = "Influence of hyperparameters on reconstruction error (testing)"
        fig = plot_multiple_runs(x_values=np.array(self._training_steps_before_testing[0]),
                                 y_values=mse_testing,
                                 labels=ExperimentTemplateBase.parameters_to_string(self._topology_parameters_list),
                                 xlabel=xlabel,
                                 ylabel=ylabel,
                                 title=title,
                                 figsize=figsize,
                                 )
        add_fig_to_doc(fig, path.join(self._docs_folder, to_safe_name(title)), doc)

        title = "Weak classifier accuracy"
        fig = plot_multiple_runs(x_values=np.array(self._training_steps_before_testing[0]),
                                 y_values=np.array(self._weak_classifier_results),
                                 labels=ExperimentTemplateBase.parameters_to_string(self._topology_parameters_list),
                                 xlabel="steps",
                                 ylabel="accuracy",
                                 title=title,
                                 figsize=figsize,
                                 hide_legend=True
                                 )
        add_fig_to_doc(fig, path.join(self._docs_folder, to_safe_name(title)), doc)

        title = "Memory * reconstruction tradeoff (testing)"
        fig = plot_multiple_runs(x_values=np.array(self._training_steps_before_testing[0]),
                                 y_values=error_memory_ratio.numpy(),
                                 labels=ExperimentTemplateBase.parameters_to_string(self._topology_parameters_list),
                                 xlabel=xlabel,
                                 ylabel="Mean Reconstruction Error times required meogabyte of memory",
                                 title=title,
                                 figsize=figsize,
                                 hide_legend=True
                                 )
        add_fig_to_doc(fig, path.join(self._docs_folder, to_safe_name(title)), doc)

        title = "Memory * error (= 1 - acc) tradeoff"
        fig = plot_multiple_runs(x_values=np.array(self._training_steps_before_testing[0]),
                                 y_values=accuracy_memory_ratio.numpy(),
                                 labels=ExperimentTemplateBase.parameters_to_string(self._topology_parameters_list),
                                 xlabel=xlabel,
                                 ylabel="Mean Reconstruction Error times required meogabyte of memory",
                                 title=title,
                                 figsize=figsize,
                                 hide_legend=True
                                 )
        add_fig_to_doc(fig, path.join(self._docs_folder, to_safe_name(title)), doc)

        title = "Entropy and mean reconstruction error"
        fig = plot_multiple_runs(x_values=np.array(self._code_entropy),
                                 y_values=mse_testing,
                                 labels=ExperimentTemplateBase.parameters_to_string(self._topology_parameters_list),
                                 xlabel="entropy",
                                 ylabel="mean reconstruction error",
                                 title=title,
                                 figsize=figsize,
                                 hide_legend=True
                                 )
        add_fig_to_doc(fig, path.join(self._docs_folder, to_safe_name(title)), doc)

        title = "Mean reconstruction error and classifier accuracy"
        fig = plot_multiple_runs(x_values=mse_testing,
                                 y_values=np.array(self._weak_classifier_results),
                                 labels=ExperimentTemplateBase.parameters_to_string(self._topology_parameters_list),
                                 xlabel="mean reconstruction error",
                                 ylabel="accuracy",
                                 title=title,
                                 figsize=figsize,
                                 hide_legend=True
                                 )
        add_fig_to_doc(fig, path.join(self._docs_folder, to_safe_name(title)), doc)

        title = "Entropy and classifier accuracy"
        fig = plot_multiple_runs(x_values=np.array(self._code_entropy),
                                 y_values=np.array(self._weak_classifier_results),
                                 labels=ExperimentTemplateBase.parameters_to_string(self._topology_parameters_list),
                                 xlabel="entropy",
                                 ylabel="accuracy",
                                 title=title,
                                 figsize=figsize,
                                 hide_legend=True
                                 )
        add_fig_to_doc(fig, path.join(self._docs_folder, to_safe_name(title)), doc)

        doc.write_file(path.join(self._docs_folder, f"{self._topology_class.__name__}_" + date + ".html"))

        print(self._memory_used)

