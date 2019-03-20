import math

from torchsim.core.eval.metrics.sp_convergence_metrics import average_sp_delta
from torchsim.core.eval.node_accessors.random_number_accessor import RandomNumberNodeAccessor
from torchsim.core.nodes import random_number_node
from torchsim.core.nodes.dataset_mnist_node import DatasetMNISTNode
from torchsim.core.eval.node_accessors.mnist_node_accessor import MnistNodeAccessor
from torchsim.core.eval.node_accessors.sp_node_accessor import SpatialPoolerFlockNodeAccessor
from torchsim.core.model import Model
from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode
from torchsim.research.experiment_templates.sp_learning_convergence_template import SpLearningConvergenceTopologyAdapter
from torchsim.research.research_topics.rt_1_1_1_one_expert_sp.topologies.mnist_sp_topology import MnistSpTopology


class MnistSpLearningConvergenceTopologyAdapter(SpLearningConvergenceTopologyAdapter):
    """SpLearningConvergenceExperiment: a subject containing MNIST and SpatialPooler Nodes."""
    _mnist: DatasetMNISTNode
    _sp: SpatialPoolerFlockNode
    _baseline: random_number_node

    def set_topology(self, topology: MnistSpTopology):
        self._topology = topology
        self._mnist = self._topology.node_mnist
        self._sp = self._topology.node_sp
        self._baseline = self._topology.node_random
        return

    def get_label_id(self) -> int:
        return MnistNodeAccessor.get_label_id(self._mnist)

    def get_learned_model_output_id(self) -> int:
        return SpatialPoolerFlockNodeAccessor.get_output_id(self._sp)

    def get_baseline_output_id(self) -> int:
        return RandomNumberNodeAccessor.get_output_id(self._baseline)

    def get_title(self) -> str:
        return 'Mutual information of SP outputs and MNIST labels'

    def get_topology(self) -> Model:
        return self._topology

    def get_device(self) -> str:
        return self._topology.device

    def get_model_output_size(self) -> int:
        return self._topology.output_dimension

    def get_average_boosting_duration(self) -> float:
        # TODO not implemented for the MNIST experiment, will be added if needed
        return 0

    def get_average_delta(self) -> float:
        delta = average_sp_delta(SpatialPoolerFlockNodeAccessor.get_sp_deltas(self._sp))
        if delta > 0:  # avoid math.domain error
            delta = math.log(delta)
        return delta
