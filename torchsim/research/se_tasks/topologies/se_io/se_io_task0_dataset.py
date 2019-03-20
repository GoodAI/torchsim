from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.core.nodes.dataset_se_objects_node import DatasetSeObjectsParams, DatasetSeObjectsNode, DatasetConfig, \
    DatasetSeObjectsOutputs
from torchsim.research.se_tasks.topologies.se_io.se_io_base import SeIoBase


class SeIoTask0Dataset(SeIoBase):
    """Access to the Task0 dataset."""

    def is_in_training_phase(self) -> bool:
        return not bool(self.get_testing_phase())

    _params: DatasetSeObjectsParams
    node_se_dataset: DatasetSeObjectsNode
    outputs: DatasetSeObjectsOutputs

    def __init__(self, params: DatasetSeObjectsParams = DatasetSeObjectsParams(dataset_config=DatasetConfig.TRAIN_TEST,
                                                                               save_gpu_memory=True)):
        self._params = params

    def _create_and_add_nodes(self):
        self.node_se_dataset = DatasetSeObjectsNode(self._params)

        # common IO
        self.outputs = self.node_se_dataset.outputs
        self.inputs = self.node_se_dataset.inputs

    def _add_nodes(self, target_group: NodeGroupBase):
        for node in [self.node_se_dataset]:
            target_group.add_node(node)

    def _connect_nodes(self):
        pass

    def get_num_labels(self):
        return self.node_se_dataset.label_size()

    def get_image_numel(self):
        return self._params.dataset_dims[0] * self._params.dataset_dims[1] * 3

    def get_image_width(self):
        return self._params.dataset_dims[1]

    def get_image_height(self):
        return self._params.dataset_dims[0]

    def get_task_id(self) -> float:
        """Constant here, see below. If set to -1, it means that the experiment should end."""
        if self.node_se_dataset.is_train_test_ended():
            return -1.0
        return 0.0

    def get_task_instance_id(self) -> float:
        """The same as task_status, not used here. If set to -1, it means that the experiment should end."""
        if self.node_se_dataset.is_train_test_ended():
            return -1.0
        return 0.0

    def get_task_status(self) -> float:
        """At the end of the task (tells if solved), not used here. -1 indicates end of entire experiment."""
        if self.node_se_dataset.is_train_test_ended():
            return -1.0
        return 0.0

    def get_task_instance_status(self) -> float:
        """After one object passes computes your performance, not used in the dataset."""
        return 0.0

    def get_reward(self) -> float:
        return 0.0

    def get_testing_phase(self) -> float:
        value = 0.0 if self.node_se_dataset.is_training() else 1.0
        return value
