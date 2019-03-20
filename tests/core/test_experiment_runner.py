import logging
import threading
from typing import Callable

import pytest

from torchsim.core.eval2.basic_experiment_template import BasicExperimentTemplate, BasicTopologyFactory
from torchsim.core.eval2.experiment_controller import ExperimentController
from torchsim.core.eval2.experiment_runner_params import ExperimentParams
from torchsim.core.eval2.measurement_manager import RunMeasurementManager, MeasurementManager
from torchsim.core.eval2.single_experiment_run import SingleExperimentManager
from torchsim.core.graph import Topology
from torchsim.core.experiment_runner import UiExperimentRunner, RunState
from torchsim.gui.observer_system import ObserverSystem
from torchsim.gui.observer_system_void import ObserverSystemVoid


class TopologyStub(Topology):
    def __init__(self, should_crash: Callable[..., bool]):
        super().__init__('cpu')
        self._should_crash = should_crash

    def step(self):
        if self._should_crash():
            raise RuntimeError

        super().step()


def run_task(f):
    thread = threading.Thread(target=f)
    thread.daemon = False
    thread.start()


class TestUiExperimentRunner:
    @staticmethod
    def crashing_step():
        raise ValueError("Crash")

    @pytest.yield_fixture(autouse=True)
    def init_test(self):
        ObserverSystem.initialized = False
        yield

    def test_state(self, caplog):
        caplog.set_level(logging.DEBUG)
        observer_system = ObserverSystemVoid()

        topology_crashing = False

        def should_crash():
            return topology_crashing

        topology_params = {'should_crash': should_crash}
        params = ExperimentParams(max_steps=0)

        # single_run = SingleExperimentManager(TopologyStub(should_crash), controller, topology_params,
        #                                      run_measurement_manager, params.create_run_params())
        template = BasicExperimentTemplate("Template")
        topology_factory = BasicTopologyFactory(TopologyStub)
        measurement_manager = MeasurementManager(None, None)
        manager = SingleExperimentManager(template, topology_factory, topology_params, measurement_manager,
                                          params.create_run_params())

        single_run = manager.create_run()

        runner = UiExperimentRunner(observer_system)

        runner.init_run(single_run)

        assert RunState.STOPPED == runner._state
        runner.start()
        assert RunState.RUNNING == runner._state
        runner.pause()
        assert RunState.PAUSED == runner._state
        runner.start()
        assert RunState.RUNNING == runner._state
        runner.stop()
        assert RunState.STOPPED == runner._state
        topology_crashing = True
        runner.step()
        assert RunState.CRASHED == runner._state

        runner.wait()

    def test_state_corner_cases(self):
        observer_system = ObserverSystemVoid()

        topology_parameters = {'device': 'cpu'}
        params = ExperimentParams(max_steps=0)

        template = BasicExperimentTemplate("Template")
        topology_factory = BasicTopologyFactory(Topology)
        measurement_manager = MeasurementManager(None, None)
        manager = SingleExperimentManager(template, topology_factory, topology_parameters, measurement_manager,
                                          params.create_run_params())
        run = manager.create_run()

        simulation = UiExperimentRunner(observer_system)

        simulation.init_run(run)
        simulation.start()
        simulation.start()
        simulation.pause()
        simulation.pause()
        simulation.stop()
        simulation.stop()
        simulation.pause()
        simulation.pause()
        assert RunState.STOPPED == simulation._state
        simulation.step()
        simulation.step()
        simulation.start()
        simulation.stop()
