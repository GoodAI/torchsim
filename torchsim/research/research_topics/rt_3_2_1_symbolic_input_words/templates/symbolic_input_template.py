import logging
from typing import List

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval2.experiment_controller import ExperimentController, ExperimentComponent
from torchsim.core.eval2.experiment_template_base import ExperimentTemplateBase, TTopology
from torchsim.core.eval2.measurement_manager import MeasurementManager, RunMeasurementManager
from torchsim.core.graph import Topology
from torchsim.research.research_topics.rt_3_2_1_symbolic_input_words.topologies.symbolic_input_words_topology import \
    SymbolicInputWordsTopology

logger = logging.getLogger(__name__)


class SpatialPoolerClusterForceSetter(ExperimentComponent):
    step_count: int = 0

    def __init__(self, topology: SymbolicInputWordsTopology):
        super().__init__()
        self.topology = topology

    def after_topology_step(self):
        self.step_count += 1
        # logger.info(f'Step {self.step_count}')
        if self.step_count == 1:
            logger.info(f'Setting SP clusters')
            self.topology.init_sp_clusters()


class SymbolicInputTemplate(ExperimentTemplateBase[SymbolicInputWordsTopology]):
    def setup_controller(self, topology: SymbolicInputWordsTopology, controller: ExperimentController,
                         run_measurement_manager: RunMeasurementManager):
        sp_cluster_setter = SpatialPoolerClusterForceSetter(topology)
        controller.register(sp_cluster_setter)

    def publish_results(self, document: Document, docs_folder: str, measurement_manager: MeasurementManager,
                        topology_parameters: List[str]):
        pass
