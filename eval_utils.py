import argparse
import gc
import logging
import sys
from contextlib import contextmanager
from typing import Callable, Any, Dict, Optional

import torch
from tqdm import tqdm

from torchsim.core.eval.series_plotter import get_stamp
from torchsim.core.eval2.basic_experiment_template import BasicExperimentTemplate, BasicTopologyFactory
from torchsim.core.eval2.experiment import Experiment
from torchsim.core.eval2.experiment_runner_params import ExperimentParams
from torchsim.core.eval2.measurement_manager import MeasurementManager
from torchsim.core.eval2.single_experiment_run import SingleExperimentManager
from torchsim.core.experiment_runner import UiExperimentRunner, BasicExperimentRunner
from torchsim.core.graph import Topology
from torchsim.core.logging import setup_logging_no_ui
from torchsim.core.logging.ui import LogObservable, setup_logging_ui
from torchsim.gui.observer_system import ObserverSystem
from torchsim.gui.observer_system_browser import ObserverSystemBrowser
logger = logging.getLogger(__name__)


def create_observer_system(storage_file: str = "observers.yaml"):
    """Factory for the observer system"""
    return ObserverSystemBrowser(
        update_period=0.1,  # Update period [s]
        storage_file=storage_file  # observers persistence storage
    )


def create_non_persisting_observer_system():
    """Not serialized, model properties not rewritten by deserialized values"""
    return ObserverSystemBrowser(
        update_period=0.1,
        storage_file=None
    )


@contextmanager
def observer_system_context(storage_file: str = "observers.yaml", persisting: bool = True,
                            log_file: Optional[str] = None):
    if persisting:
        observer_system = create_observer_system(storage_file)
    else:
        observer_system = create_non_persisting_observer_system()

    setup_logging_ui(LogObservable(), observer_system, filename=log_file)

    yield observer_system
    observer_system.stop()


def run_just_model(model, gui: bool = False, max_steps=sys.maxsize, persisting_observer_system: bool = True,
                   auto_start: bool = True):
    if gui:
        with observer_system_context("observers.yaml", persisting=persisting_observer_system) as observer_system:
            run_topology_with_ui(model, observer_system=observer_system, max_steps=max_steps, auto_start=auto_start)
    else:
        run_topology(model, max_steps, auto_start)


def parse_test_args():
    parser = argparse.ArgumentParser()
    add_test_args(parser)
    args = parser.parse_args()
    return args


def add_test_args(parser: argparse.ArgumentParser):
    parser.add_argument("--run-gui", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--clear", action="store_true", default=False)
    parser.add_argument("--computation-only", action="store_true", default=False)
    parser.add_argument("--alternative-results-folder", default=None)
    parser.add_argument("--filter-params", nargs='+', type=int, default=None)
    parser.add_argument("--show-plots", action="store_true", default=False)


def filter_params(args, params):
    if args.filter_params:
        return [params[i] for i in args.filter_params]
    else:
        return params


# progress bar in console
progress_bar = tqdm


def garbage_collect():
    gc.collect()
    torch.cuda.empty_cache()


def _create_basic_run_manager(topology_factory: Callable[..., Topology],
                              topology_params: Dict[str, Any],
                              seed: Optional[int] = None,
                              max_steps: int = 0,
                              save_model_after_run: bool = True):
    template = BasicExperimentTemplate("Template")
    topology_factory = BasicTopologyFactory(topology_factory)
    measurement_manager = MeasurementManager(None, None)

    run_params = ExperimentParams(max_steps, seed=seed, calculate_statistics=False,
                                  save_models_after_run=save_model_after_run)

    return SingleExperimentManager(template,
                                   topology_factory,
                                   topology_params,
                                   measurement_manager,
                                   run_params.create_run_params())


def _run_in_ui(observer_system: ObserverSystem, run_manager: SingleExperimentManager, auto_start: bool = True):
    runner = UiExperimentRunner(observer_system)
    runner.init_run(run_manager.create_run())
    if auto_start:
        runner.start()

    runner.wait()


def run_topology(topology: Topology,
                 seed: Optional[int] = None,
                 max_steps: int = 0,
                 auto_start: bool = True,
                 save_model_after_run: bool = True):

    def topology_factory():
        topology.stop()
        return topology

    run_topology_factory(topology_factory, {}, seed, max_steps, auto_start, save_model_after_run)


def run_topology_with_ui(topology: Topology,
                         seed: Optional[int] = None,
                         max_steps: int = 0,
                         auto_start: bool = True,
                         observer_system: Optional[ObserverSystem] = None):

    def topology_factory():
        topology.stop()
        return topology

    run_topology_factory_with_ui(topology_factory, {}, seed, max_steps, auto_start, observer_system)


def run_topology_factory(topology_factory: Callable[..., Topology],
                         topology_params: Dict[str, Any],
                         seed: Optional[int] = None,
                         max_steps: int = 0,
                         auto_start: bool = True,
                         save_model_after_run: bool = True):
    run_manager = _create_basic_run_manager(topology_factory, topology_params, seed, max_steps, save_model_after_run)

    runner = BasicExperimentRunner()
    runner.init_run(run_manager.create_run())
    if auto_start:
        runner.start()

    runner.wait()


def run_topology_factory_with_ui(topology_factory: Callable[..., Topology],
                                 topology_params: Dict[str, Any],
                                 seed: Optional[int] = None,
                                 max_steps: int = 0,
                                 auto_start: bool = True,
                                 observer_system: Optional[ObserverSystem] = None):
    run_manager = _create_basic_run_manager(topology_factory, topology_params, seed, max_steps)

    if observer_system is not None:
        _run_in_ui(observer_system, run_manager, auto_start)
    else:
        with observer_system_context() as observer_system:
            _run_in_ui(observer_system, run_manager, auto_start)


def run_experiment(experiment: Experiment):
    setup_logging_no_ui(filename=experiment.setup_logging_path(get_stamp()))
    runner = BasicExperimentRunner()
    experiment.run(runner)


def run_experiment_with_ui(experiment: Experiment, auto_start: bool = True):
    with observer_system_context() as observer_system:
        log_filename = experiment.setup_logging_path(get_stamp())
        setup_logging_ui(LogObservable(), observer_system, log_filename)
        runner = UiExperimentRunner(observer_system)
        experiment.run(runner, auto_start=auto_start)
