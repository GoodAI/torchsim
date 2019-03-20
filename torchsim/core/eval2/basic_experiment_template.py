from typing import List, Dict, Any, Callable

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval2.experiment_controller import ExperimentController
from torchsim.core.eval2.experiment_template_base import ExperimentTemplateBase, TTopology, TopologyFactory
from torchsim.core.eval2.measurement_manager import MeasurementManager, RunMeasurementManager


class BasicExperimentTemplate(ExperimentTemplateBase[TTopology]):
    def setup_controller(self, topology: TTopology, controller: ExperimentController,
                         run_measurement_manager: RunMeasurementManager):
        pass

    def publish_results(self, document: Document, docs_folder: str, measurement_manager: MeasurementManager,
                        topology_parameters: List[str]):
        pass


class BasicTopologyFactory(TopologyFactory[TTopology]):
    def __init__(self, topology: Callable[..., TTopology]):
        self._factory_method = topology

    def get_default_parameters(self) -> Dict[str, Any]:
        return {}

    def create_topology(self, **kwargs) -> TTopology:
        return self._factory_method(**kwargs)
