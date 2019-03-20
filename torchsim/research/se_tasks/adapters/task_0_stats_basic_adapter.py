from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.adapters.se_dataset_ta_running_stats_adapter import \
    SeDatasetTaRunningStatsAdapter
from torchsim.research.se_tasks.adapters.task_stats_adapter import TaskStatsAdapter
from torchsim.research.se_tasks.topologies.se_task0_basic_topology import SeT0BasicTopology


class Task0StatsBasicAdapter(TaskStatsAdapter, SeDatasetTaRunningStatsAdapter):
    """The same as the base class, but also provides aux SE stats."""

    _topology: SeT0BasicTopology

    def get_task_id(self) -> float:
        return self._topology.se_io.get_task_id()

    def get_task_instance_id(self) -> float:
        return self._topology.se_io.get_task_instance_id()

    def get_task_status(self) -> float:
        return self._topology.se_io.get_task_status()

    def get_task_instance_status(self) -> float:
        return self._topology.se_io.get_task_instance_status()

    def get_reward(self) -> float:
        return self._topology.se_io.get_reward()

    def get_testing_phase(self) -> float:
        return self._topology.se_io.get_testing_phase()

    def switch_learning(self, learning_on: bool):
        self._topology.switch_learning(learning_on)
