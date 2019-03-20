import torch

from torchsim.core import FLOAT_NEG_INF
from torchsim.core.nodes.space_engineers_connector_node import SpaceEngineersConnectorOutputs
from torchsim.research.se_tasks.topologies.se_io.se_io_base import SeIoBase
from torchsim.research.se_tasks.topologies.se_io.se_io_task0_dataset import SeIoTask0Dataset


class SeIoAccessor:
    """Accessor to particular values in the SeIoBase (SE or a dataset)."""

    @staticmethod
    def get_label_id(io_base: SeIoBase) -> int:
        """Returns the id of the current label.

        There is a difference between SE and SEDataset,
        where the dataset returns ground_truth even during testing (to allow evaluation during testing)

        Args:
            io_base: io to be used

        Returns:
            ID of the current label (ground truth in case of SEDataset).
        """
        # label of the object in Task0 (-INF if not supported in the dataset)
        if not SeIoAccessor._is_label_id_used(io_base):
            return FLOAT_NEG_INF

        # TODO: in case that there is a dataset, we want to have the label_id available even during testing
        if isinstance(io_base, SeIoTask0Dataset):
            label_tensor = io_base.outputs.task_to_agent_label_ground_truth.tensor
        else:
            label_tensor = io_base.outputs.task_to_agent_label.tensor

        _, arg_max = label_tensor.max(0)
        return int(arg_max.to('cpu').item())

    @staticmethod
    def get_label_tensor(io_base: SeIoBase) -> torch.Tensor:
        return io_base.outputs.task_to_agent_label.tensor

    @staticmethod
    def task_to_agent_label_ground_truth(io_base: SeIoBase) -> torch.Tensor:
        """This works fully only for the SE dataset, since the SE does not provide the ground truth during testing.

        Args:
            io_base: io to be used

        Returns:
            Ground truth tensor if available.
        """
        if type(io_base) is SeIoTask0Dataset:
            io_dataset: SeIoTask0Dataset = io_base
            return io_dataset.outputs.task_to_agent_label_ground_truth.tensor
        else:
            return io_base.outputs.task_to_agent_label.tensor

    @staticmethod
    def get_num_labels(io_base: SeIoBase) -> int:
        return io_base.get_num_labels()

    @staticmethod
    def _is_label_id_used(io_base: SeIoBase) -> bool:
        label = io_base.outputs.task_to_agent_label.tensor
        if label.numel() == 1 and label[0] == FLOAT_NEG_INF:
            return False
        return True

    @staticmethod
    def get_landmark_id_int(outputs: SpaceEngineersConnectorOutputs) -> int:
        # current landmark in the Task1 (-INF for the dataset)
        value = outputs.task_to_agent_location_int.tensor.to('cpu').item()
        if value == FLOAT_NEG_INF:
            return FLOAT_NEG_INF
        return int(value)

    @staticmethod
    def is_in_training_phase(io_base: SeIoBase) -> bool:
        return io_base.is_in_training_phase()

