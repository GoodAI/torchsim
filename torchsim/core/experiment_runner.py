import math
import os
import time
from abc import abstractmethod
from threading import Event, Lock, Thread

from collections import deque

from datetime import datetime
from enum import Enum, auto
from functools import partial
from typing import List, Set, Callable, Generic, TypeVar, Tuple, Optional
import logging

from torchsim.core.eval2.single_experiment_run import SingleExperimentRun
from torchsim.core.global_settings import GlobalSettings, SimulationThreadRunner
from torchsim.core.graph import Topology
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver
from torchsim.gui.observables import LambdaPropertiesObservable, ObserverPropertiesItemState, PropertiesObservable, \
    ObserverPropertiesBuilder, ObserverPropertiesItemSourceType
from torchsim.gui.observer_view import ObserverView
from torchsim.gui.observer_system import ObserverPropertiesItem, ObserverSystem, TextObservable
from torchsim.utils.node_utils import recursive_get_nodes
from torchsim.utils.os_utils import last_exception_as_html

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CleverTimer(Generic[T]):
    timer = time.perf_counter

    _value: int
    _timestamp: float

    def __init__(self, init_value: T):
        self._value = init_value
        self._timestamp = CleverTimer.timer()

    def tick(self, stored_value: T) -> Tuple[float, T]:
        prev_t, prev_v = self._timestamp, self._value
        self._timestamp = CleverTimer.timer()
        self._value = stored_value
        return self._timestamp - prev_t, prev_v

    def time_since_last_tick(self) -> float:
        return CleverTimer.timer() - self._timestamp

    def set_value(self, value: T):
        self._value = value


class FpsHelper:
    data: deque

    def __init__(self, window_length: int):
        self.data = deque(maxlen=window_length)

    def add_measurement(self, duration: float, n_steps: int):
        self.data.append((duration, n_steps))

    def calculate_avg_fps(self):
        if self.data:
            total_duration, total_iterations = map(sum, zip(*self.data))
            return FpsHelper.to_fps(total_duration, total_iterations)
        else:
            return 0

    def get_last_fps(self):
        if self.data:
            duration, iterations = self.data[-1]
            return FpsHelper.to_fps(duration, iterations)
        else:
            return 0

    def reset(self):
        self.data.clear()

    @staticmethod
    def to_fps(elapsed_time: float, n_steps: int = 1):
        try:
            return n_steps / elapsed_time
        except ZeroDivisionError:
            return 0


class SimulationMonitor(TextObservable):
    """A monitor keeping the statistics of the simulation."""
    _observer_system: ObserverSystem
    system_start: datetime
    fps_time_window: float
    fps_helper: FpsHelper
    simulation_step: int
    timer: CleverTimer

    def __init__(self, observer_system: ObserverSystem):
        self._observer_system = observer_system
        self.system_start = datetime.now()

        self.fps_time_window = 0.1  # in seconds
        average_fps_time_window = 5  # in seconds
        self.fps_helper = FpsHelper(math.ceil(average_fps_time_window / self.fps_time_window))

        self.simulation_step = 0
        self.timer = CleverTimer(self.simulation_step)

        self._observer_system.register_observer('Status', self)

    def get_data(self):

        fps = self.fps_helper.get_last_fps()
        fps_avg = self.fps_helper.calculate_avg_fps()
        fps_fmt = '.2f'
        data_to_display = {
            'Start time': self.system_start.strftime('%H:%M:%S'),
            'Uptime': str(datetime.now() - self.system_start).split('.', 2)[0],
            'Simulation step': self.simulation_step,
            'FPS': f'{fps:{fps_fmt}}',
            'FPS avg': f'{fps_avg:{fps_fmt}}',
        }

        text = '<table>'
        for key, value in data_to_display.items():
            text += f'<tr><td class="text-right" style="padding-right: 10px">{key}:</td><td>{value}</td></tr>'
        text += '</table>'
        return text

    def notify_simulation_start(self):
        self.timer.tick(self.simulation_step)

    def notify_simulation_step(self):
        self.simulation_step += 1

        if self.timer.time_since_last_tick() >= self.fps_time_window:
            elapsed_time, prev_sim_step = self.timer.tick(self.simulation_step)
            performed_steps = self.simulation_step - prev_sim_step
            self.fps_helper.add_measurement(elapsed_time, performed_steps)

    def notify_simulation_stop(self):
        self.fps_helper.reset()
        self.simulation_step = 0
        self.timer.set_value(self.simulation_step)


class GraphObservable(PropertiesObservable):
    graph: Topology
    observer_system: ObserverSystem
    visible_nodes: Set[str]
    _prop_builder: ObserverPropertiesBuilder

    def __init__(self, graph: Topology, observer_system: ObserverSystem,
                 observer_update: Callable[[str, bool], None]):
        self.graph = graph
        self.observer_system = observer_system
        self._prop_builder = ObserverPropertiesBuilder()
        self._observer_update = observer_update
        self.visible_nodes = set()

    @property
    def name(self):
        return 'Graph'

    def get_properties(self) -> List[ObserverPropertiesItem]:

        def update_node_visible(node_name: str, val: bool) -> bool:
            self._observer_update(node_name, val)
            if val:
                self.visible_nodes.add(node_name)
            else:
                # During persistence deserialization remove is called even on uninitialized visible_nodes set
                if node_name in self.visible_nodes:
                    self.visible_nodes.remove(node_name)

            return val

        prop_list = []
        prop_list.append(self._prop_builder.collapsible_header("Observers", False))
        for node in recursive_get_nodes(self.graph):
            name = node.name_with_id
            value = name in self.visible_nodes
            # noinspection PyTypeChecker
            prop_list.append(self._prop_builder.checkbox(name, value, partial(update_node_visible, name)))

        return [*prop_list, *self.graph.get_properties()]


class ExperimentRunner:
    def run(self, run: SingleExperimentRun):
        """Initialize the run, start and then wait for the run to finish (blocking).

        Args:
            run: The run to simulate.
        """
        self.init_run(run)
        self.start()
        self.wait()

    @abstractmethod
    def init_run(self, run: SingleExperimentRun):
        """Initializes the runner with a run.

        The running is later initiated via the start() method.
        """
        raise NotImplementedError

    @abstractmethod
    def start(self):
        """Starts the run.

        The run might be done asynchronously, so it's possible that this method will return sooner than the run is
        finished. Use the wait() method to wait for the run to finish.
        """
        raise NotImplementedError

    @abstractmethod
    def wait(self):
        """Waits until the run is finished."""
        pass


class RunState(Enum):
    STOPPED = 1,
    PAUSED = 2,
    RUNNING = 3,
    CRASHED = 4


class ExperimentRunnerException(Exception):
    pass


class BasicExperimentRunner(ExperimentRunner):
    def init_run(self, run: SingleExperimentRun):
        """Starts the experiment run.

        Note that if the experiment does not have a stopping condition (max steps or a component-based one), this
        will run forever.
        """
        self._run = run

    def start(self):
        """This runner works synchronously, so the run is finished when this method returns."""
        while True:
            try:
                self._run.step()
            except StopIteration:
                break

    def wait(self):
        """This runner is synchronous, there's not need to wait here."""
        return


class RunResult(Enum):
    SUCCESS = auto()
    FAILURE = auto()


class UiExperimentRunner(ExperimentRunner):
    """The main simulation class.

    This handles the starting/stopping/stepping of the simulation.
    """

    observer_system: ObserverSystem
    crashed: bool
    _observer_views: List[ObserverView]
    _simulation_monitor: SimulationMonitor
    _prop_builder: ObserverPropertiesBuilder
    _state: RunState
    _running_lock: Lock
    _step_delay: Optional[int]
    _run_queue: List[Callable]
    _run: SingleExperimentRun

    # _signals: SimulationSignals

    # @property
    # def signals(self) -> SimulationSignals:
    #     return self._signals

    @property
    def state(self) -> RunState:
        return self._state

    @property
    def step_delay(self) -> Optional[int]:
        return self._step_delay

    @step_delay.setter
    def step_delay(self, value: Optional[int]):
        self._step_delay = value

    def __init__(self, observer_system: ObserverSystem):
        # self._signals = SimulationSignals()

        # TODO: Fix logging
        # setup_logging_ui(LogObservable(), observer_system, self.experiment.setup_logging_path(get_stamp()))

        self._run = None
        self._state = RunState.STOPPED
        self._step_delay = 0
        self._running_lock = Lock()
        self.observer_system = observer_system
        self._observer_views = []
        self._graph_observable = None
        self._prop_builder = ObserverPropertiesBuilder(self, source_type=ObserverPropertiesItemSourceType.CONTROL)

        self.observer_system.register_observer('Simulation Control',
                                               LambdaPropertiesObservable(
                                                   lambda: self._get_simulation_control_observer()))

        # required to be here, so that UI deserializes the observers based on names correctly
        self._simulation_monitor = SimulationMonitor(self.observer_system)
        self._run_queue = []

        # An event object which releases the run() method when the run finishes or is restarted.
        self._run_event = Event()
        # The run is a success unless it fails.
        self._result = RunResult.SUCCESS

        SimulationThreadRunner.instance().set_simulation(self)

    def init_run(self, run: SingleExperimentRun):
        logger.info(f"Initializing run")
        self._run = run
        self._graph_observable = GraphObservable(self._run.topology, self.observer_system, self.update_node_observer)
        self.observer_system.register_observer(self._graph_observable.name, self._graph_observable)
        self._reset_observers()

    def wait(self):
        self._run_event.wait()
        self._run_event.clear()

    def _cleanup_run(self):
        if self._run is not None:
            self.observer_system.unregister_observer(self._graph_observable.name)
            self._graph_observable = None

            self.init_run(self._run.restart())

    def _get_simulation_control_observer(self):
        def st(condition: bool) -> ObserverPropertiesItemState:
            return ObserverPropertiesItemState.ENABLED if condition else ObserverPropertiesItemState.DISABLED

        # Not used - commenting out.
        # button_state = ObserverPropertiesItemState.ENABLED
        #
        # if self._state == SimulationState.CRASHED:
        #     button_state = ObserverPropertiesItemState.DISABLED

        return [
            self._prop_builder.prop('State', type(self).state, parser=None, formatter=lambda s: s.name,
                                    state=ObserverPropertiesItemState.READ_ONLY),
            self._prop_builder.button('Run', self.start,
                                      state=st(self._state == RunState.STOPPED or
                                               self._state == RunState.PAUSED)),
            self._prop_builder.button('Pause', self.pause,
                                      state=st(self._state == RunState.RUNNING)),
            self._prop_builder.button('Step', self.step,
                                      state=st(self._state == RunState.STOPPED or
                                               self._state == RunState.PAUSED)),
            self._prop_builder.button('Stop', self.stop,
                                      state=st(self._state != RunState.STOPPED)),
            self._prop_builder.button('Finish run', self.finish_run,
                                      state=st(self._state != RunState.STOPPED),
                                      hint='Stop and finish experiment run'),

            self._prop_builder.button('Save all tensors', self.save_graph, caption='Save'),
            self._prop_builder.button('Load all tensors', self.load_graph, caption='Load'),
            self._prop_builder.button('Node persistence', self.observer_system.load_model_values, caption='Load',
                                      hint='Load persisted values of model properties'),
            self._prop_builder.button('Node persistence', self.observer_system.save_model_values, caption='Save',
                                      hint='Save values of model properties'),
            *self._global_settings_properties()
        ]

    def _global_settings_properties(self):
        gs = GlobalSettings.instance()

        def update_memory_block_min_size(value: int):
            GlobalSettings.instance().observer_memory_block_minimal_size = value

        return [
            self._prop_builder.collapsible_header('Global Settings', False),
            self._prop_builder.number_int("Memory block observer minimal size", gs.observer_memory_block_minimal_size,
                                          update_memory_block_min_size),
            self._prop_builder.auto("Step delay [ms]", type(self).step_delay)
        ]

    def update_node_observer(self, name: str, show: bool):
        view = self._get_view(name)
        if show:
            node = next((node for node in recursive_get_nodes(self._run.topology) if node.name_with_id == name), None)
            if node is not None:
                view.set_observables(node.get_observables())
        else:
            self._observer_views.remove(view)
            view.close()

    def _reset_observers(self):
        for name in [view.name for view in self._observer_views]:
            self.update_node_observer(name, False)

        for node_name in self._graph_observable.visible_nodes:
            self.update_node_observer(node_name, True)

    def _get_view(self, name) -> ObserverView:
        found_view = None
        for view in self._observer_views:
            if name in view.name:
                found_view = view
                break

        if found_view is None:
            found_view = ObserverView(f'Node observers - {name}', self.observer_system,
                                      strip_observer_name_prefix=f'{name}.')
            self._observer_views.append(found_view)

        return found_view

    def start(self):
        if self._state == RunState.RUNNING:
            logger.info("Run already started, new start requested and ignored")
            return

        logger.info("Starting the run loop")

        self._simulation_monitor.notify_simulation_start()
        self._state = RunState.RUNNING

        assert self._running_lock.locked() is False, 'Running lock should be free'
        self._running_lock.acquire()
        thread = Thread(target=self._simulation_loop)
        thread.daemon = False
        thread.start()

    def stop(self):
        self._state = RunState.STOPPED
        self._running_lock.acquire()
        self._cleanup_run()
        self._simulation_monitor.notify_simulation_stop()
        self._result = RunResult.FAILURE
        self._running_lock.release()

    def finish_run(self):
        self._run.stop()
        self.stop()
        self._run_event.set()

    def pause(self):
        if self._state != RunState.RUNNING:
            return

        self._state = RunState.PAUSED
        self._running_lock.acquire()
        self._running_lock.release()

    @staticmethod
    def _log_last_exception():
        logger.error(last_exception_as_html())

    def _do_step(self):
        self._simulation_monitor.notify_simulation_step()
        self._run.step()

    def step(self):
        if self._state == RunState.RUNNING:
            self.pause()
        else:
            self._state = RunState.RUNNING
            # noinspection PyBroadException
            try:
                self._do_step()
                self._state = RunState.PAUSED
            except StopIteration:
                logger.debug("The run is stopping correctly - end condition reached during step")
                self._run_event.set()
            except Exception:
                self._notify_simulation_crashed()

    def _simulation_loop(self):
        logger.debug("Starting simulation loop")
        while self._state == RunState.RUNNING:
            # noinspection PyBroadException
            try:
                self._execute_run_queue()
                self._do_step()
                if self.step_delay is not None:
                    time.sleep(self.step_delay / 1000.0)
            except StopIteration:
                logger.debug("The run is stopping correctly - end condition reached during simulation loop")
                self._state = RunState.STOPPED
                self._simulation_monitor.notify_simulation_stop()
                self._cleanup_run()
                self._run_event.set()
                break
            except Exception:
                self._notify_simulation_crashed()
        logger.debug("Exiting simulation loop")
        self._running_lock.release()

    def _get_persistence_location(self):
        return os.path.join(os.getcwd(), 'data', 'stored', type(self._run.topology).__name__)

    def save_graph(self):
        if self._run.topology is None:
            logger.warning('The topology has not been initialized yet, cannot save. Do a step.')
            return

        path = self._get_persistence_location()

        logger.info("Graph saving started..")
        saver = Saver(path)
        self._run.topology.save(saver)
        saver.save()

        logger.info('Graph saved.')

    def load_graph(self):
        if self._run.topology is None:
            logger.warning('The topology has not been initialized yet, cannot load. Do a step.')
            return

        path = self._get_persistence_location()
        if not os.path.exists(path):
            logger.warning(f"There is no saved model at location {path}")
            return

        logger.info("Graph loading started..")
        loader = Loader(path)
        try:
            self._run.topology.load(loader)
        except FileNotFoundError:
            logger.exception(f"Loading of model failed")

        logger.info('Graph loaded.')

    def is_running(self) -> bool:
        return self._state == RunState.RUNNING

    def _notify_simulation_crashed(self):
        self._log_last_exception()
        self._state = RunState.CRASHED
        self._result = RunResult.FAILURE
        self._run_event.set()

    def add_to_run_queue(self, runnable: Callable):
        self._run_queue.append(runnable)

    def _execute_run_queue(self):
        for runnable in self._run_queue:
            runnable()
        self._run_queue.clear()
