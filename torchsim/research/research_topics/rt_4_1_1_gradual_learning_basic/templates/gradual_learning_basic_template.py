import os
import numpy as np
import logging
from typing import List, cast

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval.series_plotter import plot_multiple_runs, add_fig_to_doc, to_safe_name
from torchsim.core.eval2.experiment_controller import ExperimentController, ExperimentComponent
from torchsim.core.eval2.experiment_template_base import ExperimentTemplateBase
from torchsim.core.eval2.measurement_manager import MeasurementManager, RunMeasurementManager
from torchsim.core.eval2.run_measurement import SingleRunMeasurements
from torchsim.research.research_topics.rt_4_1_1_gradual_learning_basic.topologies.gradual_learning_basic_topology import \
    GradualLearningBasicTopology, NotForgettingExperimentParams, GLExperimentParams

logger = logging.getLogger(__name__)

ACCURACY_1 = 'accuracy_1'
ACCURACY_PER_FLOCK_1 = 'accuracy_per_flock_1'
ACCURACY_SINGLE_1 = 'accuracy_single_1'
ACCURACY_PER_FLOCK_SINGLE_1 = 'accuracy_per_flock_single_1'
ACCURACY_2 = 'accuracy_2'
ACCURACY_PER_FLOCK_2 = 'accuracy_per_flock_2'
ACCURACY_SINGLE_2 = 'accuracy_single_2'
ACCURACY_PER_FLOCK_SINGLE_2 = 'accuracy_per_flock_single_2'
DATASET_2_SEQUENCE_ID = 'dataset_2_sequence_id'


def compute_moving_average(data: np.array, window_size: int) -> np.array:
    return np.convolve(data, np.ones((window_size,))/window_size, mode='same')


def plot_testing_accuracy(steps: np.ndarray, d1: np.ndarray, d2: np.ndarray, window_size: int, name: str,
                          trim_size: int, labels: List[str], document: Document, docs_folder: str):
    a1 = compute_moving_average(d1, window_size)
    a2 = compute_moving_average(d2, window_size)
    title = f'{name}, window size: {window_size}'

    f = plot_multiple_runs(
        np.expand_dims(steps[trim_size:-trim_size - 1], axis=0),
        np.stack((
            np.expand_dims(a1[trim_size:-trim_size - 1], axis=0),
            np.expand_dims(a2[trim_size:-trim_size - 1], axis=0)),
            axis=-1),
        title=title,
        ylabel='accuracy',
        xlabel='steps',
        labels=labels,
        other_params=[{'color': None}]
    )
    add_fig_to_doc(f, os.path.join(docs_folder, title), document)


class GradualLearningBasicMeasuringComponent(ExperimentComponent):
    def __init__(self,
                 topology: GradualLearningBasicTopology,
                 run_measurement_manager: RunMeasurementManager):
        self._topology = topology
        self._run_measurement_manager = run_measurement_manager

        run_measurement_manager.add_measurement_f(ACCURACY_SINGLE_1, lambda: topology.get_accuracy_single_1())
        run_measurement_manager.add_measurement_f(ACCURACY_PER_FLOCK_SINGLE_1,
                                                  lambda: topology.get_accuracy_per_flock_single_1())
        run_measurement_manager.add_measurement_f(ACCURACY_1, lambda: topology.get_accuracy_1())
        run_measurement_manager.add_measurement_f(ACCURACY_PER_FLOCK_1, lambda: topology.get_accuracy_per_flock_1())

        run_measurement_manager.add_measurement_f(ACCURACY_SINGLE_2, lambda: topology.get_accuracy_single_2())
        run_measurement_manager.add_measurement_f(ACCURACY_PER_FLOCK_SINGLE_2,
                                                  lambda: topology.get_accuracy_per_flock_single_2())
        run_measurement_manager.add_measurement_f(ACCURACY_2, lambda: topology.get_accuracy_2())
        run_measurement_manager.add_measurement_f(ACCURACY_PER_FLOCK_2, lambda: topology.get_accuracy_per_flock_2())

        run_measurement_manager.add_measurement_f(DATASET_2_SEQUENCE_ID, lambda: topology.get_actual_sequence_ids())

    def calculate_run_results(self):
        super().calculate_run_results()

        # measurements = self._run_measurement_manager.measurements
        # measurements.add_custom_data('custom', 40)


class StepCountingExperimentComponent(ExperimentComponent):
    step_count: int = 0

    def __init__(self):
        super().__init__()

    def after_topology_step(self):
        self.step_count += 1


class SpatialPoolerClusterForceSetter(StepCountingExperimentComponent):
    def __init__(self, topology: GradualLearningBasicTopology):
        super().__init__()
        self.topology = topology

    def after_topology_step(self):
        super().after_topology_step()
        if self.step_count == 1:
            logger.info(f'Initializing SP clusters to symbolic dataset symbols')
            self.topology.init_sp_clusters()


class LearnForgetTestExperimentComponentBase(StepCountingExperimentComponent):
    def __init__(self, topology: GradualLearningBasicTopology, params: NotForgettingExperimentParams):
        super().__init__()
        self.topology = topology
        self.params = params

    def should_end_run(self) -> bool:
        return self.step_count == self.params.phase_1_steps + self.params.phase_2_steps + self.params.phase_3_steps

class NotForgettingExperimentComponent(LearnForgetTestExperimentComponentBase):
    def after_topology_step(self):
        super().after_topology_step()
        if self.step_count == 1:
            logger.info(f'Starting phase 1, step: {self.step_count}')
            self.topology.set_sequences_filter(0, [True, True, False] * 2)
            self.topology.set_sequences_filter(1, [True, True, True] * 2)
            self.topology.active_dataset = 0
        if self.step_count == self.params.phase_1_steps:
            logger.info(f'Starting phase 2, step: {self.step_count}')
            self.topology.active_dataset = 1
            self.topology.set_sequences_filter(1, [True, True, False] * 2)
        if self.step_count == self.params.phase_1_steps + self.params.phase_2_steps:
            logger.info(f'Starting phase 3, step: {self.step_count}')
            self.topology.active_dataset = 1
            self.topology.set_sequences_filter(1, [True, True, True] * 2)


    @classmethod
    def publish_results(cls, document: Document, docs_folder: str, measurement_manager: MeasurementManager,
                        topology_parameters: List[str]):
        experiment_params = measurement_manager.single_run_measurements[0].model_parameters['params']['experiment_params']
        gl_params = cast(GLExperimentParams, experiment_params)
        not_forgetting_params = cast(NotForgettingExperimentParams, gl_params.params)

        start_of_testing_phase = not_forgetting_params.phase_1_steps + not_forgetting_params.phase_2_steps
        start_of_testing_phase -= 1000  # start 500 steps backwards to initialize moving average

        steps = np.array(measurement_manager.get_values_from_all_runs('current_step'))[0, start_of_testing_phase:]
        sequence_ids = np.array(measurement_manager.get_values_from_all_runs(DATASET_2_SEQUENCE_ID))[0, start_of_testing_phase:]
        accuracy_per_flock_1_single = np.array(measurement_manager.get_values_from_all_runs(ACCURACY_PER_FLOCK_SINGLE_1))[0, start_of_testing_phase:]
        accuracy_per_flock_2_single = np.array(measurement_manager.get_values_from_all_runs(ACCURACY_PER_FLOCK_SINGLE_2))[0, start_of_testing_phase:]

        labels = topology_parameters

        # title = 'sequence_id'
        # f = plot_multiple_runs(
        #     np.expand_dims(steps, axis=0),
        #     np.expand_dims(sequence_id, axis=0),
        #     title=title,
        #     ylabel='sequence_id',
        #     xlabel='steps',
        #     labels=labels
        # )
        # add_fig_to_doc(f, os.path.join(docs_folder, title), document)

        d1 = np.mean(accuracy_per_flock_1_single, axis=-1)
        d2 = np.mean(accuracy_per_flock_2_single, axis=-1)

        plot_testing_accuracy(steps, d1, d2, window_size=180, name='accuracy_testing_1_2', trim_size=180,
                              labels=labels, document=document, docs_folder=docs_folder)
        seq_filter = (sequence_ids == 2) | (sequence_ids == 5)
        # seq_filter_inv = (sequence_id != 2) & (sequence_id != 5)

        # plot_testing_accuracy(
        #     # np.arange(np.sum(seq_filter)),
        #     steps[seq_filter],
        #     d1[seq_filter],
        #     d2[seq_filter],
        #     window_size=180, name='accuracy_just_sequences_3,6',
        #     trim_size=180,
        #     labels=labels, document=document, docs_folder=docs_folder
        # )
        # plot_testing_accuracy(
        #     steps[~seq_filter],
        #     d1[~seq_filter],
        #     d2[~seq_filter],
        #     window_size=180, name='accuracy_just_sequences_1,2,4,5',
        #     trim_size=180,
        #     labels=labels, document=document, docs_folder=docs_folder
        # )


class OneShotLearningExperimentComponent(LearnForgetTestExperimentComponentBase):
    def after_topology_step(self):
        super().after_topology_step()
        if self.step_count == 1:
            logger.info(f'Starting phase 1, step: {self.step_count}')
            self.topology.set_sequences_filter(0, [True, True, False] * 2)
            self.topology.set_sequences_filter(1, [True, True, True, True, True, False])
            self.topology.active_dataset = 0
        if self.step_count == self.params.phase_1_steps + self.params.phase_2_steps:
            logger.info(f'Starting phase 3, step: {self.step_count}')
            self.topology.active_dataset = 1
            self.topology.set_sequences_filter(1, [True, True, True] * 2)

    @classmethod
    def publish_results(cls, document: Document, docs_folder: str, measurement_manager: MeasurementManager,
                        topology_parameters: List[str]):
        experiment_params = measurement_manager.single_run_measurements[0].model_parameters['params']['experiment_params']
        gl_params = cast(GLExperimentParams, experiment_params)
        not_forgetting_params = cast(NotForgettingExperimentParams, gl_params.params)

        start_of_testing_phase = not_forgetting_params.phase_1_steps + not_forgetting_params.phase_2_steps
        start_of_testing_phase -= 1000  # start 500 steps backwards to initialize moving average

        steps = np.array(measurement_manager.get_values_from_all_runs('current_step'))[0, start_of_testing_phase:]
        sequence_ids = np.array(measurement_manager.get_values_from_all_runs(DATASET_2_SEQUENCE_ID))[0, start_of_testing_phase:]
        accuracy_per_flock_1_single = np.array(measurement_manager.get_values_from_all_runs(ACCURACY_PER_FLOCK_SINGLE_1))[0, start_of_testing_phase:]
        accuracy_per_flock_2_single = np.array(measurement_manager.get_values_from_all_runs(ACCURACY_PER_FLOCK_SINGLE_2))[0, start_of_testing_phase:]

        labels = topology_parameters

        d1 = np.mean(accuracy_per_flock_1_single, axis=-1)
        d2 = np.mean(accuracy_per_flock_2_single, axis=-1)

        plot_testing_accuracy(steps, d1, d2, window_size=180, name='accuracy_testing_1_2', trim_size=180,
                              labels=labels, document=document, docs_folder=docs_folder)
        seq_filter_3_6 = (sequence_ids == 2) | (sequence_ids == 5)
        seq_filter_6 = (sequence_ids == 5)

        # plot_testing_accuracy(
        #     steps[seq_filter_3_6],
        #     d1[seq_filter_3_6],
        #     d2[seq_filter_3_6],
        #     window_size=180, name='accuracy_just_sequences_3,6',
        #     trim_size=180,
        #     labels=labels, document=document, docs_folder=docs_folder
        # )
        # plot_testing_accuracy(
        #     steps[seq_filter_6],
        #     d1[seq_filter_6],
        #     d2[seq_filter_6],
        #     window_size=180, name='accuracy_just_sequences_6',
        #     trim_size=180,
        #     labels=labels, document=document, docs_folder=docs_folder
        # )


class KnowledgeReuseExperimentComponent(LearnForgetTestExperimentComponentBase):
    def after_topology_step(self):
        super().after_topology_step()
        if self.step_count == 1:
            logger.info(f'Starting phase 1, step: {self.step_count}')
            self.topology.set_sequences_filter(1, [True, False, False] * 2)
            self.topology.active_dataset = 1
        if self.step_count == self.params.phase_1_steps:
            logger.info(f'Starting phase 2, step: {self.step_count}')
            self.topology.active_dataset = 0
            self.topology.set_sequences_filter(0, [True, True, True] * 2)
            self.topology.set_sequences_filter(1, [True, True, False] * 2)
        if self.step_count == self.params.phase_1_steps + self.params.phase_2_steps:
            logger.info(f'Starting phase 3, step: {self.step_count}')
            self.topology.active_dataset = 1
            self.topology.set_sequences_filter(1, [True, True, True] * 2)


    @classmethod
    def publish_results(cls, document: Document, docs_folder: str, measurement_manager: MeasurementManager,
                        topology_parameters: List[str]):
        pass


class GradualLearningBasicTemplate(ExperimentTemplateBase[GradualLearningBasicTopology]):
    def setup_controller(self, topology: GradualLearningBasicTopology, controller: ExperimentController,
                         run_measurement_manager: RunMeasurementManager):
        sp_cluster_setter = SpatialPoolerClusterForceSetter(topology)
        controller.register(sp_cluster_setter)

        measuring_component = GradualLearningBasicMeasuringComponent(topology, run_measurement_manager)
        controller.register(measuring_component)

        experiment_params = topology.params.experiment_params
        if experiment_params is not None:
            experiment_component = experiment_params.component(topology, experiment_params.params)
            controller.register(experiment_component)

    def publish_results(self, document: Document, docs_folder: str, measurement_manager: MeasurementManager,
                        topology_parameters: List[str]):
        """An alternative to the _publish_results method, this is called from _publish_results now

        Draw and add your plots to the document here.
        """

        steps = measurement_manager.get_values_from_all_runs('current_step')
        accuracy_1 = measurement_manager.get_values_from_all_runs(ACCURACY_1)
        accuracy_per_flock_1 = measurement_manager.get_values_from_all_runs(ACCURACY_PER_FLOCK_1)
        accuracy_2 = measurement_manager.get_values_from_all_runs(ACCURACY_2)
        accuracy_per_flock_2 = measurement_manager.get_values_from_all_runs(ACCURACY_PER_FLOCK_2)

        multiple_flocks_alpha = 0.05
        # measurement_manager.single_run_measurements[0].get_custom_data('custom')
        labels = topology_parameters

        a1 = np.array(accuracy_1)
        a2 = np.array(accuracy_2)

        title = 'accuracy_1_2'
        f = plot_multiple_runs(
            steps,
            np.stack((a1, a2), axis=-1),
            title=title,
            ylabel='accuracy_1_2',
            xlabel='steps',
            labels=labels,
            other_params=[{'color': None}]
        )
        add_fig_to_doc(f, os.path.join(docs_folder, title), document)

        experiment_params = measurement_manager.single_run_measurements[0].model_parameters['params']['experiment_params']
        experiment_params.component.publish_results(document, docs_folder, measurement_manager, topology_parameters)

        title = 'accuracy_per_flock_1'
        f = plot_multiple_runs(
            steps,
            accuracy_per_flock_1,
            title=title,
            ylabel='accuracy_per_flock_1',
            xlabel='steps',
            labels=labels,
            hide_legend=True,
            other_params=[{'alpha': multiple_flocks_alpha}]
        )
        add_fig_to_doc(f, os.path.join(docs_folder, title), document)

        # title = 'accuracy_2'
        # f = plot_multiple_runs(
        #     steps,
        #     accuracy_2,
        #     title=title,
        #     ylabel='accuracy_2',
        #     xlabel='steps',
        #     labels=labels
        # )
        # add_fig_to_doc(f, os.path.join(docs_folder, title), document)

        title = 'accuracy_per_flock_2'
        f = plot_multiple_runs(
            steps,
            accuracy_per_flock_2,
            title=title,
            ylabel='accuracy_per_flock_2',
            xlabel='steps',
            labels=labels,
            hide_legend=True,
            other_params=[{'alpha': multiple_flocks_alpha}]
        )
        add_fig_to_doc(f, os.path.join(docs_folder, title), document)




        # Add table with MSE single-step values
        # prediction_mse_values = [f'{v:.5f}' for v in prediction_mse[0]]
        # document.add_table(['step', 'prediction_mse'], list(zip(steps[0], prediction_mse_values)),
        #                    attribs={'style': 'font-size:0.8em;'})


        # for single_run_measurements in measurement_manager.single_run_measurements:
        #     self.publish_one_run(single_run_measurements, document, docs_folder, topology_parameters)


        doc_path = os.path.join(docs_folder, to_safe_name(self.experiment_name + ".html"))
        if doc_path.startswith("\\\\?\\"):
            doc_path = doc_path[len("\\\\?\\"):]

        # Note not logging to UI now
        logger.info(f'Results published <a href="file:///{doc_path}">{doc_path}</a>')
        logger.info(f'Results published {doc_path}')

    def publish_one_run(self, single_run_measurements: SingleRunMeasurements, document: Document, docs_folder: str,
                        topology_parameters: List[str]):

        steps = single_run_measurements.get_items('current_step')
        accuracy_1 = single_run_measurements.get_items(ACCURACY_1)
        accuracy_per_flock_1 = single_run_measurements.get_items(ACCURACY_PER_FLOCK_1)
        accuracy_2 = single_run_measurements.get_items(ACCURACY_2)
        accuracy_per_flock_2 = single_run_measurements.get_items(ACCURACY_PER_FLOCK_2)

        labels = topology_parameters

        multiple_flocks_alpha = 0.05

        title = 'accuracy_per_flock_1'
        f = plot_multiple_runs(
            [steps],
            [accuracy_per_flock_1],
            title=title,
            ylabel='accuracy_per_flock_1',
            xlabel='steps',
            labels=labels,
            hide_legend=True,
            other_params=[{'alpha': multiple_flocks_alpha}]
        )
        add_fig_to_doc(f, os.path.join(docs_folder, title), document)

        title = 'accuracy_per_flock_2'
        f = plot_multiple_runs(
            [steps],
            [accuracy_per_flock_2],
            title=title,
            ylabel='accuracy_per_flock_2',
            xlabel='steps',
            labels=labels,
            hide_legend=True,
            other_params=[{'alpha': multiple_flocks_alpha}]
        )
        add_fig_to_doc(f, os.path.join(docs_folder, title), document)

        a1 = np.array(accuracy_1)
        a2 = np.array(accuracy_2)

        # title = 'accuracy_1_2'
        # f = plot_multiple_runs(
        #     [steps],
        #     np.stack((a1, a2), axis=-1),
        #     title=title,
        #     ylabel='accuracy_1_2',
        #     xlabel='steps',
        #     labels=labels,
        #     other_params=[{'color': None}]
        # )
        # add_fig_to_doc(f, os.path.join(docs_folder, title), document)
