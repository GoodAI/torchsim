from dataclasses import dataclass
from os import path
from typing import List, Union, NamedTuple

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval.doc_generator.heading import Heading
from torchsim.core.eval.series_plotter import plot_multiple_runs
from torchsim.core.eval2.experiment_controller import ExperimentController, ExperimentComponent
from torchsim.core.eval2.experiment_template_base import ExperimentTemplateBase
from torchsim.core.eval2.measurement_manager import MeasurementManager, RunMeasurementManager
from torchsim.core.eval2.train_test_switchable import TrainTestSwitchable
from torchsim.core.models.expert_params import ParamsBase
from torchsim.topologies.gradual_learning_topology import GradualLearningTopology


@dataclass
class GradualLearningWorldParams(ParamsBase):
    initial_training_steps: int = 10000
    testing_steps: int = 100
    untraining_steps: int = 10000
    retraining_steps: int = 100
    retraining_phases: int = 100

    every_second_is_baseline: bool = False

    experiment_name: str = ""


class GradualLearningWorldComponent(ExperimentComponent):

    # TODO WIP does not work yet

    """
    Initial training -> initial testing -> untraining ... untraining -> repeat(testing -> retraining)
    """
    def __init__(self, topology: Union[GradualLearningTopology, TrainTestSwitchable],
                 run_measurement_manager: RunMeasurementManager,
                 initial_training_steps, testing_steps, untraining_steps, retraining_steps, retraining_phases):
        self.topology = topology
        self.run_measurement_manager = run_measurement_manager
        self.initial_training_steps = initial_training_steps
        self.testing_steps = testing_steps
        self.untraining_steps = untraining_steps
        self.retraining_steps = retraining_steps
        self.retraining_phases = retraining_phases

        run_measurement_manager.add_measurement_f("initial training error", topology.get_error, 1,
                                                  self.is_initial_training)
        run_measurement_manager.add_measurement_f("initial testing error", topology.get_error, 1,
                                                  self.is_initial_testing)
        run_measurement_manager.add_measurement_f("untraining error", topology.get_error, 1,
                                                  self.is_untraining)
        run_measurement_manager.add_measurement_f("retraining error", topology.get_error, 1,
                                                  self.is_retraining)
        run_measurement_manager.add_measurement_f("testing error", topology.get_error, 1,
                                                  self.is_testing)

        self.steps_num = 0
        self._phase = [False, False, False, False, False]

    def before_topology_step(self):
        if self.steps_num == 0:
            self.training_configuration()
            self.set_phase_table(0)
        if self.steps_num == self.initial_training_steps:
            self.testing_configuration()
            self.set_phase_table(1)
        if self.steps_num == self.initial_training_steps + self.testing_steps:
            self.untraining_configuration()
            self.set_phase_table(2)
        if self.steps_num >= self.initial_training_steps + self.testing_steps + self.untraining_steps:
            last_phase_length = self.steps_num \
                                - self.initial_training_steps - self.testing_steps - self.untraining_steps
            last_phase_mod = last_phase_length % (self.retraining_steps + self.testing_steps)
            if last_phase_mod == 0:
                self.testing_configuration()
                self.set_phase_table(3)
            if last_phase_mod == self.testing_steps:
                self.retraining_configuration()
                self.set_phase_table(4)

        self.steps_num += 1

    def set_phase_table(self, phase_nr: int):
        self._phase = [False, False, False, False, False]
        self._phase[phase_nr] = True

    def is_initial_training(self):
        return self._phase[0]

    def is_initial_testing(self):
        return self._phase[1]

    def is_untraining(self):
        return self._phase[2]

    def is_testing(self):
        return self._phase[3]

    def is_retraining(self):
        return self._phase[4]

    def training_configuration(self):
        self.topology.switch_input_to(0)
        self.topology.switch_to_training()

    def testing_configuration(self):
        self.topology.switch_input_to(1)
        self.topology.switch_to_testing()

    def untraining_configuration(self):
        self.topology.switch_input_to(2)
        self.topology.switch_to_training()

    def retraining_configuration(self):
        self.topology.switch_input_to(3)
        self.topology.switch_to_training()

    def should_end_run(self) -> bool:
        return self.steps_num >= self.initial_training_steps + self.testing_steps + self.untraining_steps \
               + (self.retraining_steps + self.testing_steps) * self.retraining_phases


class GradualLearningWorldTemplate(ExperimentTemplateBase[Union[GradualLearningTopology, TrainTestSwitchable]]):
    def __init__(self, template_params: GradualLearningWorldParams,
                 experiment_name="GlNnExperiment",
                 **params):
        super().__init__(experiment_name=experiment_name, **params)
        self.template_params = template_params

    def setup_controller(self, topology: Union[GradualLearningTopology, TrainTestSwitchable],
                         controller: ExperimentController,
                         run_measurement_manager: RunMeasurementManager):
        controller.register(GradualLearningWorldComponent(
            topology,
            run_measurement_manager,
            initial_training_steps=self.template_params.initial_training_steps,
            testing_steps=self.template_params.testing_steps,
            untraining_steps=self.template_params.untraining_steps,
            retraining_steps=self.template_params.retraining_steps,
            retraining_phases=self.template_params.retraining_phases))

    def publish_results(self, document: Document, docs_folder: str, measurement_manager: MeasurementManager,
                        topology_parameters: List[str]):
        document.add(Heading(self.template_params.experiment_name))
        for title in measurement_manager.single_run_measurements[0].get_item_names():
            if title not in ["initial training error", "initial testing error", "untraining error",
                             "retraining error", "testing error"]:
                continue
            data = measurement_manager.get_values_from_all_runs(title)
            n_runs = len(data)
            if self.template_params.every_second_is_baseline:  # TODO MS
                labels = (n_runs // 2) * ["experiment", "baseline"]
            else:
                labels = n_runs * [""]
            plot_multiple_runs(list(range(len(data[0]))),
                               data,
                               ylabel="error",
                               xlabel="steps",
                               labels=labels,
                               title=title,
                               smoothing_window_size=21,
                               path=path.join(docs_folder, title),
                               doc=document)

