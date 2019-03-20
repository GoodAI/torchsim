from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.graph import Topology
from torchsim.core.nodes import ActionMonitorNode
from torchsim.core.nodes import DatasetSeObjectsNode, DatasetSeObjectsParams, DatasetConfig


class SeDatasetObjectsTopology(Topology):
    _node_se_dataset: DatasetSeObjectsNode
    _node_action_monitor: ActionMonitorNode

    def __init__(self, params: DatasetSeObjectsParams = None):
        super().__init__(device='cuda')
        if params is None:
            self._params = DatasetSeObjectsParams(dataset_config=DatasetConfig.TRAIN_TEST,
                                                  dataset_size=SeDatasetSize.SIZE_32)
        else:
            self._params = params

        self._node_se_dataset = DatasetSeObjectsNode(self._params)

        self.add_node(self._node_se_dataset)
