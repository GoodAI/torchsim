from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.adapters.se_dataset_ta_running_stats_adapter import \
    SeDatasetTaRunningStatsAdapter
from torchsim.research.se_tasks.adapters.task_stats_adapter import TaskStatsAdapter
from torchsim.research.se_tasks.topologies.se_task1_basic_topology import SeT1Bt


class Task1StatsBasicAdapter(TaskStatsAdapter, SeDatasetTaRunningStatsAdapter):
    """Duplicates code from Task0StatsBasicAdapter."""
    # TODO DRY

    _topology: SeT1Bt

    def get_task_id(self) -> float:
        return self._topology.node_se_connector.outputs.metadata_task_id.tensor.cpu().item()

    def get_task_instance_id(self) -> float:
        return self._topology.node_se_connector.outputs.metadata_task_instance_id.tensor.cpu().item()

    def get_task_status(self) -> float:
        return self._topology.node_se_connector.outputs.metadata_task_status.tensor.cpu().item()

    def get_task_instance_status(self) -> float:
        return self._topology.node_se_connector.outputs.metadata_task_instance_status.tensor.cpu().item()

    def get_reward(self) -> float:
        return self._topology.node_se_connector.outputs.reward_output.tensor.cpu().item()

    def get_testing_phase(self) -> float:
        return self._topology.node_se_connector.outputs.metadata_testing_phase.tensor.cpu().item()

    def switch_learning(self, learning_on: bool):
        self._topology.switch_learning(learning_on)
