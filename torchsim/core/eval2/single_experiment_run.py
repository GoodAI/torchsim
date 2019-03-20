import logging
import os
from typing import Generic, Dict, Any, Optional

import tqdm

from torchsim.core.eval2.document_publisher import DocumentPublisher
from torchsim.core.eval2.experiment_controller import ExperimentController
from torchsim.core.eval2.experiment_runner_params import SingleExperimentRunParams
from torchsim.core.eval2.experiment_template_base import TTopology, TopologyFactory, ExperimentTemplateBase
from torchsim.core.eval2.measurement_manager import MeasurementManager, RunMeasurementManager
from torchsim.core.persistence.saver import Saver

logger = logging.getLogger(__name__)


class SingleExperimentManager(Generic[TTopology]):
    """A single experiment run.

    The run prepares the topology and then steps through the simulation, calling the controller's methods at the
    appropriate moments.

    TODO: Refactor this, it's kind of too convoluted and stateful.
    """

    def __init__(self,
                 template: ExperimentTemplateBase[TTopology],
                 topology_factory: TopologyFactory[TTopology],
                 topology_params: Dict[str, Any],
                 measurement_manager: MeasurementManager,
                 run_params: SingleExperimentRunParams,
                 run_idx: Optional[int] = 0):

        self._iterator = None

        self._template = template
        self._topology_factory = topology_factory
        self._topology_params = topology_params
        self.measurement_manager = measurement_manager

        self._run_params = run_params
        self._run_idx = run_idx

    def try_load(self) -> bool:
        # Try to get the run from cache (this doesn't load the data files yet).
        run_measurement_manager = self.measurement_manager.try_get_cached(self._template.experiment_name,
                                                                          self._topology_params)

        if run_measurement_manager is None:
            return False

        logger.info("Loaded from cache.")
        # If the run exists, but we don't want to calculate anything from it, just continue to the next run.

        controller = ExperimentController()
        topology = self._topology_factory.create_topology(**self._topology_params)
        self._template.setup_controller(topology, controller, run_measurement_manager)

        run_measurement_manager.load_from_cache()

        if self._run_params.calculate_statistics:
            controller.calculate_run_results()
            # TODO: Add caching of this as well.
            self.measurement_manager.add_results(run_measurement_manager)

        return True

    def create_run(self) -> 'SingleExperimentRun[TTopology]':
        # Create a new run.
        run_measurement_manager = self.measurement_manager.create_new_run(
            self._template.experiment_name, self._topology_params)

        parameters_print = DocumentPublisher.parameters_to_string([self._topology_params])[0]
        logger.info(f'Creating run with parameters: \n{parameters_print}')

        controller = ExperimentController()
        topology = self._topology_factory.create_topology(**self._topology_params)
        self._template.setup_controller(topology, controller, run_measurement_manager)

        topology.assign_ids()

        return SingleExperimentRun(self, topology, controller, self._topology_params, run_measurement_manager,
                                   self._run_params, self._run_idx)


class SingleExperimentRun(Generic[TTopology]):
    """A single experiment run.

    The run prepares the topology and then steps through the simulation, calling the controller's methods at the
    appropriate moments.

    Note: This should be rewritten, it's kind of too convoluted and stateful.
    """

    def __init__(self,
                 manager: SingleExperimentManager,
                 topology: TTopology,
                 controller: ExperimentController,
                 topology_params: Dict[str, Any],
                 run_measurement_manager: RunMeasurementManager,
                 run_params: SingleExperimentRunParams,
                 run_idx: int):

        self._manager = manager
        self.topology = topology
        self.controller = controller
        self._run_measurement_manager = run_measurement_manager
        self._iterator = None
        self._topology_params = topology_params

        self._run_idx = run_idx

        self._save_cache = run_params.save_cache
        self._calculate_statistics = run_params.calculate_statistics
        self._max_steps = run_params.max_steps
        self._save_model_after_run = run_params.save_model_after_run
        self._current_step = 0

    @staticmethod
    def _forever():
        num = 0
        while True:
            yield num
            num += 1

    def _init_loop(self):
        if self._iterator is None:
            self.topology.prepare()

            if self._max_steps > 0:
                iterator = tqdm.tqdm(range(self._max_steps), "progress: ")
            else:
                iterator = self._forever()

            self._iterator = iterator.__iter__()

    def step(self):
        """Perform one step of the experiment."""
        self._init_loop()

        try:
            self._iterator.__next__()
            # make one step of the topology
            self.controller.before_topology_step()
            self.topology.step()
            self.controller.after_topology_step()
            # make measurement
            self._run_measurement_manager.step(self._current_step)
            self._current_step += 1
            if self.controller.should_end_run():
                self._iterator.close()
        except StopIteration:
            self.stop()
            # Notify the runner that the iteration stopped here.
            raise

    def restart(self) -> 'SingleExperimentRun[TTopology]':
        if self._iterator is not None:
            self._iterator.close()
        return self._manager.create_run()

    def stop(self):
        # No more steps, clean up.
        self._iterator = None

        if self._save_cache:
            self._run_measurement_manager.save_cache()

        if self._save_model_after_run:
            path = os.path.join(self._run_measurement_manager.cache_folder, 'saved_models')
            saver = Saver(path)
            self.topology.save(saver, self.topology.name + f"_{self._run_idx}")
            saver.save()

        if self._calculate_statistics:
            self.controller.calculate_run_results()

        self._manager.measurement_manager.add_results(self._run_measurement_manager)
