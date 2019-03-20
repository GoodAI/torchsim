from typing import List

import pytest

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval2.basic_experiment_template import BasicTopologyFactory
from torchsim.core.eval2.experiment_controller import ExperimentController, ExperimentComponent, \
    TrainTestComponentParams, TrainTestControllingComponent
from torchsim.core.eval2.experiment import SingleExperimentManager
from torchsim.core.eval2.experiment_runner_params import SingleExperimentRunParams
from torchsim.core.eval2.experiment_template_base import ExperimentTemplateBase, TTopology
from torchsim.core.eval2.measurement_manager import RunMeasurementManager, MeasurementManager
from torchsim.core.eval2.run_measurement import TrainTestMeasurementPartitioning
from torchsim.core.eval2.train_test_switchable import TrainTestSwitchable
from torchsim.core.experiment_runner import BasicExperimentRunner
from torchsim.core.graph import Topology
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver


class TopologyStub(Topology):
    pass


class ComponentStub(ExperimentComponent):
    def __init__(self, idx: int, events: list, should_end: bool = False):
        self.idx = idx
        self._events = events
        self._name = f'component {self.idx}'
        self._should_end = should_end

    def event(self, event_name):
        return self._name, event_name

    def before_topology_step(self):
        super().before_topology_step()
        self._events.append(self.event('before_step'))

    def after_topology_step(self):
        super().after_topology_step()
        self._events.append(self.event('after_step'))

    def should_end_run(self) -> bool:
        return self._should_end

    def calculate_run_results(self):
        super().calculate_run_results()
        self._events.append(self.event('finish'))


def test_experiment_controller_chain():
    controller = ExperimentController()

    events = []

    component1 = controller.register(ComponentStub(1, events))
    component2 = controller.register(ComponentStub(2, events))

    assert [component1, component2] == controller._components

    controller.before_topology_step()
    controller.after_topology_step()
    controller.before_topology_step()
    controller.after_topology_step()
    controller.calculate_run_results()

    assert [('component 1', 'before_step'),
            ('component 2', 'before_step'),
            ('component 1', 'after_step'),
            ('component 2', 'after_step'),
            ('component 1', 'before_step'),
            ('component 2', 'before_step'),
            ('component 1', 'after_step'),
            ('component 2', 'after_step'),
            ('component 1', 'finish'),
            ('component 2', 'finish')] == events


@pytest.mark.parametrize('should_end_1,should_end_2', [
    (False, False),
    (False, True),
    (True, False),
    (True, True)
])
def test_experiment_controller_should_end(should_end_1, should_end_2):
    controller = ExperimentController()

    events = []

    controller.register(ComponentStub(1, events, should_end_1))
    controller.register(ComponentStub(2, events, should_end_2))

    assert (should_end_1 or should_end_2) == controller.should_end_run()


class TrainTestTopologyStub(Topology, TrainTestSwitchable):
    def save(self, parent_saver: Saver):
        pass

    def load(self, parent_loader: Loader):
        pass

    _value = 0

    def switch_to_training(self):
        self._value = 1

    def switch_to_testing(self):
        self._value = 2

    def value(self):
        return self._value


class TemplateStub(ExperimentTemplateBase[TrainTestTopologyStub]):
    def __init__(self, train_test_params: TrainTestComponentParams):
        super().__init__(train_test_params)
        self._train_test_params = train_test_params
        self.value = 0

    def setup_controller(self, topology: TTopology, controller: ExperimentController,
                         run_measurement_manager: RunMeasurementManager):
        train_test_component = TrainTestControllingComponent(topology, run_measurement_manager, self._train_test_params)
        controller.register(train_test_component)

        train_test_component.add_measurement_f_training('item', self.measure)
        train_test_component.add_measurement_f_training('model_item', topology.value)
        train_test_component.add_measurement_f_testing('item', self.measure)
        train_test_component.add_measurement_f_testing('model_item', topology.value)

    def measure(self):
        self.value += 1
        return self.value

    def publish_results(self, document: Document, docs_folder: str, measurement_manager: MeasurementManager,
                        topology_parameters: List[str]):
        pass


def test_train_test_experiment():
    topology_params = {'device': 'cpu'}

    train_test_params = TrainTestComponentParams(num_testing_phases=2, num_testing_steps=2,
                                                 overall_training_steps=4)
    run_params = SingleExperimentRunParams(load_cache=False, save_cache=False, calculate_statistics=True,
                                           max_steps=train_test_params.max_steps,
                                           save_model_after_run=False)
    template = TemplateStub(train_test_params)

    topology_factory = BasicTopologyFactory(lambda device: TrainTestTopologyStub(device))

    measurement_manager = MeasurementManager(None, None)

    # single_run = SingleExperimentManager(topology, controller, topology_params, run_measurement_manager, run_params)
    single_run = SingleExperimentManager(template, topology_factory, topology_params, measurement_manager, run_params)

    runner = BasicExperimentRunner()
    runner.run(single_run.create_run())

    run_measurements = measurement_manager.single_run_measurements[0]
    assert all(len(item) > 0 for item in run_measurements._values)

    partitioning = TrainTestMeasurementPartitioning(run_measurements)
    training_phases = partitioning.partition_to_training_phases('item')
    training_phases_model = partitioning.partition_to_training_phases('model_item')
    testing_phases = partitioning.partition_to_testing_phases('item')
    testing_phases_model = partitioning.partition_to_testing_phases('model_item')

    assert [(0, [(0, 1), (1, 2)]), (1, [(4, 5), (5, 6)])] == training_phases
    assert [(0, [(0, 1), (1, 1)]), (1, [(4, 1), (5, 1)])] == training_phases_model
    assert [(0, [(2, 3), (3, 4)]), (1, [(6, 7), (7, 8)])] == testing_phases
    assert [(0, [(2, 2), (3, 2)]), (1, [(6, 2), (7, 2)])] == testing_phases_model
