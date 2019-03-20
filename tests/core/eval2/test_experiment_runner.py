import os
from typing import List, Dict, Any

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval2.experiment_controller import ExperimentController, ExperimentComponent
from torchsim.core.eval2.experiment import Experiment
from torchsim.core.eval2.experiment_runner_params import ExperimentParams
from torchsim.core.eval2.experiment_template_base import ExperimentTemplateBase, TopologyFactory
from torchsim.core.eval2.measurement_manager import MeasurementManager, RunMeasurementManager
from torchsim.core.experiment_runner import BasicExperimentRunner
from torchsim.core.graph import Topology
from tests.testing_utils import TEMPORARY_TEST_PATH


class TopologyStub(Topology):
    def __init__(self, device: str, param):
        super().__init__(device)
        self.param = param

    def get_something(self):
        return self.param


class ExperimentComponentStub(ExperimentComponent):
    def __init__(self, topology, events):
        self._topology = topology
        self._events = events

    def before_topology_step(self):
        self._events.append('do_before_topology_step')

    def after_topology_step(self):
        self._events.append('do_after_topology_step')

    def should_end_run(self) -> bool:
        self._events.append(f'should_end_run: {self._topology.get_something()}')
        return False


class ExperimentTemplateStub(ExperimentTemplateBase[TopologyStub]):
    def __init__(self, events):
        super().__init__('Experiment Stub')
        self._events = events

    def setup_controller(self, topology: TopologyStub, controller: ExperimentController,
                         run_measurement_manager: RunMeasurementManager):
        controller.register(ExperimentComponentStub(topology, self._events))

    def publish_results(self, document: Document, docs_folder: str, measurement_manager: MeasurementManager,
                        topology_parameters: List[str]):
        pass


def test_basic_experiment_template_run():
    events = []

    template = ExperimentTemplateStub(events)

    params = ExperimentParams(max_steps=2, save_cache=False, load_cache=False, calculate_statistics=False,
                              experiment_folder=TEMPORARY_TEST_PATH)

    topology_parameters = [
        {'param': 1},
        {'param': 2}
    ]

    topology_factory = TopologyFactoryStub()

    experiment = Experiment(template, topology_factory, topology_parameters, params)

    runner = BasicExperimentRunner()
    experiment.run(runner)
    doc_path = experiment.results_path

    expected = []

    for ps in topology_parameters:
        value = ps['param']
        expected += ['do_before_topology_step', 'do_after_topology_step', f'should_end_run: {value}'] * 2

    assert expected == events
    assert doc_path is None


class ExperimentComponentMeasuringStub(ExperimentComponent):
    def __init__(self, topology: TopologyStub, run_measurement_manager: RunMeasurementManager):
        run_measurement_manager.add_measurement_f('foo', topology.get_something)


class ExperimentTemplateMeasuringStub(ExperimentTemplateBase[TopologyStub]):
    def setup_controller(self, topology: TopologyStub, controller: ExperimentController,
                         run_measurement_manager: RunMeasurementManager):
        controller.register(ExperimentComponentMeasuringStub(topology, run_measurement_manager))

    def publish_results(self, document: Document, docs_folder: str, measurement_manager: MeasurementManager,
                        topology_parameters: List[str]):
        pass


class TopologyFactoryStub(TopologyFactory[TopologyStub]):
    def get_default_parameters(self) -> Dict[str, Any]:
        return {'param': None}

    def create_topology(self, param=None) -> TopologyStub:
        return TopologyStub('cpu', param)


def test_experiment_runner():
    topology_factory = TopologyFactoryStub()

    template = ExperimentTemplateMeasuringStub("Measuring template test")

    topology_parameters = [{'param': 42}]
    params = ExperimentParams(max_steps=3, save_cache=False, load_cache=False, calculate_statistics=True,
                              experiment_folder=TEMPORARY_TEST_PATH)

    experiment = Experiment(template, topology_factory, topology_parameters, params)

    runner = BasicExperimentRunner()
    experiment.run(runner)

    measurements = experiment.measurement_manager.single_run_measurements[0]
    assert [(0, 42), (1, 42), (2, 42)] == measurements.get_step_item_tuples('foo')
    assert experiment.results_path is not None
    assert os.path.exists(experiment.results_path)
