import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchsim.core.actions import SpaceEngineersActionsDescriptor
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.utils.baselines_utils import ObservationStorage, output_size
from torchsim.core.nodes.nn_node import NNetParams, NNetNode
from torchsim.core.nodes.space_engineers_connector_node import SpaceEngineersConnectorNode
from torchsim.utils.seed_utils import set_global_seeds
from torchsim.utils.space_engineers_connector import SpaceEngineersConnectorConfig

# put custom NNet params here, overriding default ones
_nn_node_params = {
    'mixed_mode': True, # whether cpu/gpu use should be mixed
    # 'curriculum': (3000, -1)  # custom curriculum TODO: for debugging
}


class NNet(nn.Module):
    """Basic CNN model from https://goo.gl/ehssLf"""

    # Loss function
    criterion = nn.CrossEntropyLoss()

    def __init__(self, input_shape=(3, 64, 64), output_shape=20):
        """Constructor for a basic CNN architecture.

        Args:
            input_shape: shape of input fed into the network
            output_shape: shape of output produced by network
        """
        super(NNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        h, w = output_size(output_size(output_size(output_size(
            input_shape[1:], 5), 2, 2), 5), 2, 2)
        self.fc1 = nn.Linear(h*w*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_shape)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NNetTopology(Topology):
    """Topology for training any neural network."""

    _current_step: int

    def __init__(self, **kwargs):
        super().__init__(device='cpu')

        self._current_step = 0

        # set topology params and configs
        self._params = NNetParams(NNetParams.default_params())
        self._params.set_params(_nn_node_params)  # params defined in this file
        self._params.set_params(kwargs)  # params defined in GUI

        # SE config and setup
        self._se_config = SpaceEngineersConnectorConfig()
        self._se_config.curriculum = list(self._params.curriculum)
        self._actions_descriptor = SpaceEngineersActionsDescriptor()
        # set SE specific params automatically
        self._params.set_params({
            'input_shape': (3, self._se_config.render_width, self._se_config.render_height),
            'output_size': self._se_config.agent_to_task_buffer_size
        })

        # observation storage params
        self._observation_types = {
            'x': (self._params.buffer_size, *self._params.input_shape),  # observations
            'y': (self._params.buffer_size, self._params.output_size),  # labels
        }

        # data storage
        self._storage = ObservationStorage(
            self._params.buffer_size,
            self._observation_types)
        self._storage.to('cpu' if self._params.mixed_mode else self.device)

        # network needs to have the global seeds to have set before creating (outside of the node in this case)
        set_global_seeds(seed=self._params.seed)

        # ==================================================
        # NOTE: Replace here with your own architecture.
        #       It needs to be able to take the correct
        #       input and output (shape/size)
        # ==================================================
        # neural network setup
        self._network = NNet(
            input_shape=self._params.input_shape,
            output_shape=self._params.output_size
        ).to('cuda' if self._params.mixed_mode else self.device)
        # ==================================================

        # neural net optimizer
        self._optimizer = optim.Adam(self._network.parameters(),
                                     lr=self._params.lr)

        # SE Node
        self._se_connector = SpaceEngineersConnectorNode(
            self._actions_descriptor,
            self._se_config)

        # NNet Node
        self._nnet_node = NNetNode(
            self._network,
            self._optimizer,
            self._storage,
            self._params,
            name='Neural Network Node')

        # add nodes to the topology
        self.add_node(self._nnet_node)
        self.add_node(self._se_connector)

        # connect it all up
        Connector.connect(
            self._se_connector.outputs.image_output,
            self._nnet_node.inputs.input)

        Connector.connect(
            self._se_connector.outputs.task_to_agent_label,
            self._nnet_node.inputs.label)

        Connector.connect(
            self._se_connector.outputs.metadata_testing_phase,
            self._nnet_node.inputs.testing_phase)

        Connector.connect(
            self._nnet_node.outputs.output,
            self._se_connector.inputs.agent_action,
            is_backward=True)

        Connector.connect(
            self._nnet_node.outputs.label,
            self._se_connector.inputs.agent_to_task_label,
            is_backward=True)

        # necessary, but not used connector
        # TODO: remove once node is not needing this
        Connector.connect(
            self._nnet_node.outputs.task_control,
            self._se_connector.inputs.task_control,
            is_backward=True)

    def step(self):
        super().step()
        self._current_step += 1