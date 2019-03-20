import torch

from torchsim.core.graph import Topology
from torchsim.research.se_tasks.adapters.task_stats_adapter import TaskStatsAdapter
from torchsim.topologies.nnet_topology import NNetTopology


class Task0BaselinesStatsBasicAdapter(TaskStatsAdapter):
    """The same as the base class, but also provides aux SE stats."""
    _topology: NNetTopology

    def get_topology(self) -> Topology:
        return self._topology

    def set_topology(self, topology: NNetTopology):
        self._topology = topology

    def get_task_id(self) -> float:
        return self._topology._se_connector.outputs.metadata_task_id.tensor.cpu().item()

    def get_task_instance_id(self) -> float:
        return self._topology._se_connector.outputs.metadata_task_instance_id.tensor.cpu().item()

    def get_task_status(self) -> float:
        return self._topology._se_connector.outputs.metadata_task_status.tensor.cpu().item()

    def get_task_instance_status(self) -> float:
        return self._topology._se_connector.outputs.metadata_task_instance_status.tensor.cpu().item()

    def get_reward(self) -> float:
        return self._topology._se_connector.outputs.reward_output.tensor.cpu().item()

    def get_testing_phase(self) -> float:
        if self._topology._se_connector.outputs.metadata_testing_phase.tensor is None:
            return False
        else:
            return self._topology._se_connector.outputs.metadata_testing_phase.tensor.cpu().item()

    def switch_learning(self, learning_on: bool):
        pass

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


# Commented out - not used anywhere, and threw warnings.
# class Task0BaselinesAdapterBase(Task0OnlineLearningAdapterBase):
#     """Provides an insight into learned SP representations in the hierarchy of FLockNodes and a random baseline."""
#
#     _topology: NNetTopology
#
#     def set_topology(self, topology: NNetTopology):
#         self._topology = topology
#
#     def get_topology(self) -> Topology:
#         return self._topology
#
#     def get_label_tensor(self):
#         return self._topology._se_connector.outputs.task_to_agent_label.tensor
#
#     def get_label_id(self) -> int:
#         _, arg_max = self.get_label_tensor().max(0)
#         return int(arg_max.to('cpu').item())
#
#     def clone_label_tensor(self) -> torch.Tensor:
#         return self.get_label_tensor().clone()
#
#     def clone_ground_truth_label_tensor(self) -> torch.Tensor:
#         pass
#
#     def clone_baseline_output_tensor_for_labels(self) -> torch.Tensor:
#         pass
#
#     def clone_random_baseline_output_tensor_for_labels(self) -> torch.Tensor:
#         pass
#
#     def clone_predicted_label_tensor_output(self) -> torch.Tensor:
#         return self._topology._nnet_node.outputs[1].tensor.clone()
#
#     def get_random_baseline_output_id_for_labels(self) -> int:
#         pass
#
#     def get_baseline_output_id_for(self, layer_id: int) -> int:
#         pass
#
#     def get_sp_output_size_for(self, layer_id: int) -> int:
#         pass
#
#     def clone_sp_output_tensor_for(self, layer_id: int) -> torch.Tensor:
#         pass
#
#     def get_sp_output_id_for(self, layer_id: int) -> int:
#         pass
#
#     def get_average_log_delta_for(self, layer_id: int) -> float:
#         pass
#
#     def get_average_boosting_duration_for(self, layer_id: int) -> float:
#         pass
#
#     def get_device(self) -> str:
#         return self._topology.device
#
#     def get_current_step(self) -> int:
#         return self._topology._current_step
#
#     def switch_learning(self, learning_on: bool):
#         pass
#
#     def dataset_switch_learning(self, learning_on: bool, just_hide_labels: bool):
#         # SE probably do not support manual switching between train/test
#         assert type(self._se_io) is SeIoTask0Dataset
#
#         io_dataset: SeIoTask0Dataset = self._se_io
#         io_dataset.node_se_dataset.switch_training(learning_on, just_hide_labels)
#
#     def is_learning(self) -> bool:
#         # return self._topology._is_learning
#         pass
#
#     def is_output_id_available_for(self, layer_id: int) -> bool:
#         """All the topology has flock_size=1"""
#         return True
#
#     def get_title(self) -> str:
#         return 'T0 - Narrow hierarchy'