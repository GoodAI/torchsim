import torch

from torchsim.core.graph import Topology
from torchsim.core.model import Model
from torchsim.research.experiment_templates.dataset_simulation_running_stats_template import \
    DatasetSeTaSimulationRunningStatsAdapter


class SeDatasetTaRunningStatsAdapter(DatasetSeTaSimulationRunningStatsAdapter):
    """
    a thing which can provide memory/fps stats
    """

    _last_step_duration: float = 0.001

    def set_topology(self, topology) -> Topology:
        self._topology = topology
        return self._topology

    def get_topology(self) -> Model:
        return self._topology

    def get_memory_allocated(self) -> float:
        return torch.cuda.memory_allocated() / (1024 ** 2)

    def get_title(self) -> str:
        return 'FPS and memory requirements for different Flock params'

    def get_max_memory_allocated(self) -> float:
        return torch.cuda.max_memory_allocated() / (1024 ** 2)

    def get_max_memory_cached(self) -> float:
        return torch.cuda.max_memory_cached() / (1024 ** 2)

    def get_memory_cached(self) -> float:
        return torch.cuda.memory_cached() / (1024 ** 2)
