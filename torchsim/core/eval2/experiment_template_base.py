from abc import ABC, abstractmethod

import logging
from typing import Generic, List, Any, Dict, TypeVar

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval2.experiment_controller import ExperimentController
from torchsim.core.eval2.measurement_manager import MeasurementManager, RunMeasurementManager
from torchsim.core.graph import Topology

logger = logging.getLogger(__name__)


TTopology = TypeVar('TTopology', bound=Topology)


class TopologyFactory(Generic[TTopology]):
    """A factory used by the runner to create topologies based on parameters.

    This also provides the default parameter values for the purpose of document publishing.

    Either subclass this and provide the default parameters manually, or look at torchsim.core.eval2.scaffolding.
    """
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def create_topology(self, **kwargs) -> TTopology:
        pass


class ExperimentTemplateBase(ABC, Generic[TTopology]):
    """A prescription for an experiment.

    The template describes how a single experiment runs given some parameters, how the measurements are collected,
    and how are they accumulated/published at the end.
    """

    @property
    def experiment_name(self):
        """The name of the experiment."""
        return self._experiment_name

    def __init__(self, experiment_name=None, **params):
        """Initialize the template.

        Args:
            experiment_name: The name of the experiment.
            params: Any additional params. These will be printed in the resulting document.
        """
        self._experiment_name = experiment_name
        self._additional_params = params

    def get_additional_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Returns any additional parameters of the experiment."""
        return self._additional_params

    @abstractmethod
    def setup_controller(self, topology: TTopology, controller: ExperimentController,
                         run_measurement_manager: RunMeasurementManager):
        """Register the components for a single experiment run with the controller."""
        pass

    @abstractmethod
    def publish_results(self, document: Document, docs_folder: str, measurement_manager: MeasurementManager,
                        topology_parameters: List[str]):
        """Publishes the results of the whole set of experiments into the document.

        The measurements of all runs are contained in the measurement_manager instance.
        """
        pass
