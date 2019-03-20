from torchsim.core.nodes.dataset_phased_se_objects_task_node import PhasedSeObjectsTaskNode, PhasedSeObjectsTaskParams
from torchsim.research.se_tasks.topologies.se_io.se_io_task0_dataset import SeIoTask0Dataset


class SeIoTask0DatasetPhased(SeIoTask0Dataset):
    """Access to the Task0 dataset."""

    _params: PhasedSeObjectsTaskParams
    node_se_dataset: PhasedSeObjectsTaskNode

    def __init__(self, params: PhasedSeObjectsTaskParams):
        super().__init__(params.dataset_params)
        self._phased_params = params

    def _create_and_add_nodes(self):
        self.node_se_dataset = PhasedSeObjectsTaskNode(self._phased_params)

        # common IO
        self.outputs = self.node_se_dataset.outputs
        self.inputs = self.node_se_dataset.inputs
