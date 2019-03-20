from torchsim.research.experiment_templates.simulation_running_stats_template import SeTaSimulationRunningStatsAdapter
from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.adapters.se_dataset_ta_running_stats_adapter import \
    SeDatasetTaRunningStatsAdapter


class SeTaRunningStatsAdapter(SeDatasetTaRunningStatsAdapter, SeTaSimulationRunningStatsAdapter):
    """
    the same as the base class, but also provides aux SE stats
    """

    def get_task_id(self) -> float:
        return self._topology._node_se_connector.outputs.metadata_task_id.tensor.cpu().item()

    def get_task_instance_id(self) -> float:
        return self._topology._node_se_connector.outputs.metadata_task_instance_id.tensor.cpu().item()

    def get_task_status(self) -> float:
        return self._topology._node_se_connector.outputs.metadata_task_status.tensor.cpu().item()

    def get_task_instance_status(self) -> float:
        return self._topology._node_se_connector.outputs.metadata_task_instance_status.tensor.cpu().item()

    def get_reward(self) -> float:
        return self._topology._node_se_connector.outputs.reward.tensor.cpu().item()

    def get_testing_phase(self) -> float:
        return self._topology._node_se_connector.outputs.metadata_testing_phase.tensor.cpu().item()
#