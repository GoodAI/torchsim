import gc
import inspect
import logging
import os

import torch
from shutil import copyfile
from typing import Generic, Dict, Any, Callable, List, Optional
import numpy as np

from torchsim.core.eval.series_plotter import get_stamp, get_experiment_results_folder, to_safe_name, to_safe_path
from torchsim.core.eval2.experiment_template_base import ExperimentTemplateBase, TTopology
from torchsim.core.eval2.experiment_runner_params import ExperimentParams
from torchsim.core.eval2.measurement_manager import MeasurementManager, MeasurementManagerParams
from torchsim.core.eval2.document_publisher import DocumentPublisher
from torchsim.core.eval2.scaffolding import TopologyFactory
from torchsim.core.eval2.single_experiment_run import SingleExperimentManager
from torchsim.core.experiment_runner import ExperimentRunner
from torchsim.core.logging import flush_logs
from torchsim.utils.seed_utils import set_global_seeds

logger = logging.getLogger(__name__)


def garbage_collect():
    gc.collect()
    torch.cuda.empty_cache()


class ExperimentRunnerException(Exception):
    pass


class Experiment(Generic[TTopology]):
    """A runner for a set of experiments."""

    def __init__(self,
                 template: ExperimentTemplateBase,
                 topology_factory: TopologyFactory[TTopology],
                 topology_parameters: List[Dict[str, Any]],
                 params: ExperimentParams,
                 measurement_manager_params: Optional[MeasurementManagerParams] = None):
        """Initializes the runner.

        Args:
            template: An experiment template.
            topology_factory: A factory that creates a topology given one item from the topology_parameters list.
                              It also provides default parameter values.
            topology_parameters: A set of parameters for the topologies that the experiment will run.
            params: The parameters for this experiment runner.
            measurement_manager_params: The parameters for the MeasurementManager.
        """

        self._template = template
        self._topology_factory = topology_factory
        self._topology_parameters = topology_parameters

        self._params = params
        self._experiment_folder = self._get_experiment_folder_path(self._params.experiment_folder)

        self._measurement_manager_params = measurement_manager_params or MeasurementManagerParams()

        self.measurement_manager = None

        self._run_params = params.create_run_params()
        self._init_seed(params.seed)

        self._timestamp_start = None

        # If not None, the log will be copied during publish_results().
        self._log_file = None

        # This stores the path to the results file produced by the whole experiment run.
        self.results_path = None

    @staticmethod
    def _get_docs_folder_path(experiment_folder: str, experiment_name: str, stamp: str):
        return to_safe_path(os.path.join(experiment_folder, "docs", to_safe_name(experiment_name + stamp)))

    def _get_experiment_folder_path(self, experiment_folder):
        if experiment_folder is None:
            return to_safe_path(
                os.path.join(get_experiment_results_folder(), to_safe_name(self._template.experiment_name)))

        return experiment_folder

    @staticmethod
    def _get_default_topology_params(topology_factory: Callable[..., TTopology]) -> Dict[str, Any]:
        """Get default params of the topology factory.

        Returns: dictionary {param_name: param_default_value}
        """
        my_params = dict(inspect.signature(topology_factory).parameters)

        params_dict = dict((key, value.default)
                           for key, value in zip(my_params.keys(), list(my_params.values()))
                           if key is not 'self')

        return params_dict

    def _init_seed(self, seed: int):
        """Determines whether these measurements will be deterministic (across different runs)."""
        self.rand = np.random.RandomState()
        self.rand.seed(seed=seed)

        set_global_seeds(seed)

    def setup_logging_path(self, timestamp) -> str:
        if not os.path.isdir(self._experiment_folder):
            os.makedirs(self._experiment_folder)

        self._log_file = os.path.join(self._experiment_folder, f'run_{timestamp}.log')

        return self._log_file

    def run(self, runner: ExperimentRunner, auto_start: bool = True):
        """A generator which performs one step of the experiment per iteration.

        One step here means one simulation step of the topology of a single experiment run. If no run exists, the next
        run in the queue of all runs is created and a step is done there. If all single experiment runs were already
        finished, the simulation is stopped and the results are generated.
        """

        self._timestamp_start = get_stamp()
        self.measurement_manager = MeasurementManager(self._experiment_folder, self._measurement_manager_params)

        if self._params.clear_cache:
            self.measurement_manager.clear_cache()
        elif self._params.load_cache:
            # This can only happen in the cache is not cleared.
            self.measurement_manager.load_cache_index()

        garbage_collect()  # empty the memory from previous experiments so that they do not influence this one
        logger.info(f"Starting experiment with {len(self._topology_parameters)} runs")

        self._run_all(runner, auto_start)

        if not self._params.calculate_statistics:
            # No results get published.
            return

        self.results_path = self.produce_results()

    def _run_all(self, runner: ExperimentRunner, auto_start: bool = True):
        for topology_idx, topology_parameters in enumerate(self._topology_parameters):
            garbage_collect()

            parameters_print = DocumentPublisher.parameters_to_string([topology_parameters])[0]
            logger.info(f'Creating run no. {topology_idx} with parameters: \n{parameters_print}')

            single_experiment_manager = SingleExperimentManager(self._template,
                                                                self._topology_factory,
                                                                topology_parameters,
                                                                self.measurement_manager,
                                                                self._run_params,
                                                                topology_idx)
            if not single_experiment_manager.try_load():
                run = single_experiment_manager.create_run()
                runner.init_run(run)
                if auto_start:
                    runner.start()

                runner.wait()

    def produce_results(self):
        logger.info(f"Processing results")
        default_topology_parameters = self._topology_factory.get_default_parameters()
        docs_folder = self._params.docs_folder or self._get_docs_folder_path(self._experiment_folder,
                                                                             self._template.experiment_name,
                                                                             self._timestamp_start)

        topology_parameters = [dict(params) for params in self._topology_parameters]
        for params in topology_parameters:
            # This is used to distinguish several runs with the same parameters if needed.
            if '__id' in params:
                del params['__id']

        document_publisher = DocumentPublisher(self._template,
                                               docs_folder,
                                               type(self._template),
                                               default_topology_parameters,
                                               topology_parameters,
                                               self._params)
        docs_file = document_publisher.publish_results(self._timestamp_start, self.measurement_manager)
        logger.info(f"Done")
        if self._log_file is not None:
            flush_logs()
            log_file_destination = os.path.join(docs_folder, 'run.log')
            copyfile(self._log_file, log_file_destination)

        return docs_file
