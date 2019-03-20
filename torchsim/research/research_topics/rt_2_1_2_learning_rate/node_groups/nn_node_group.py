import torch
import torch.optim as optim

from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_base import EmptyOutputs
from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.core.nodes import ConstantNode, SwitchNode
from torchsim.core.nodes.nn_node import NNetParams, NNetNode
from torchsim.research.research_topics.rt_2_1_2_learning_rate.adapters.modular.model_classification_adapter_base import \
    ModelClassificationAdapterBase
from torchsim.topologies.nnet_topology import NNet
from torchsim.topologies.toyarch_groups.ncm_group_base import ClassificationTaskInputs
from torchsim.utils.baselines_utils import ObservationStorage
from torchsim.utils.seed_utils import set_global_seeds

# put custom NNet params here, overriding default ones
_nn_node_params = {
    'mixed_mode': True,  # whether cpu/gpu use should be mixed
}


class NnNodeGroup(NodeGroupBase[ClassificationTaskInputs, EmptyOutputs], ModelClassificationAdapterBase):
    """ Contains the NN (baseline) designed for the Task0.
    """

    _num_labels: int
    _is_training: bool

    device: str = 'cuda'  # network accesses the self.device

    def __init__(self,
                 num_labels: int,
                 buffer_s: int,
                 batch_s: int,
                 model_seed: int,
                 lr: float,
                 num_epochs: int,
                 image_size=SeDatasetSize.SIZE_24,
                 num_channels=3):
        """
        Initialize the node group containing the NN used as a baseline for Task0
        Args:
            num_labels: num labels in the dataset (20 for the Task0)
            image_size: size of the image, 24 by default (the result is 24*24*3) then
            model_seed: used for deterministic experiments
            lr: learning rate
        """
        super().__init__("Task 0 - NN Model", inputs=ClassificationTaskInputs(self))

        # output layer size
        self._num_labels = num_labels  # the network should configure output size from here ideally
        img_size = image_size.value  # input size is 3 * img_size **2

        kwargs = {'lr': lr,
                  'buffer_size': buffer_s,
                  'batch_size': batch_s,
                  'seed': model_seed,
                  'input_shape': (num_channels, img_size, img_size),  # note: this is correct (see node.step())
                  'output_size': self._num_labels,
                  'num_epochs': num_epochs}

        # set topology params and configs
        self._params = NNetParams(NNetParams.default_params())
        self._params.set_params(_nn_node_params)  # params defined in this file
        self._params.set_params(kwargs)  # params defined in the constructor

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

        # neural network setup
        self._network = NNet(
            input_shape=self._params.input_shape,
            output_shape=self._params.output_size
        ).to('cuda' if self._params.mixed_mode else self.device)

        # neural net optimizer
        self._optimizer = optim.Adam(self._network.parameters(),
                                     lr=self._params.lr)

        # NNet Node
        self._nnet_node = NNetNode(
            self._network,
            self._optimizer,
            self._storage,
            self._params,
            name='Neural Network Node')

        self.add_node(self._nnet_node)

        # connect the input of the network
        Connector.connect(
            self.inputs.image.output,
            self._nnet_node.inputs.input
        )
        # source of targets for learning here
        Connector.connect(
            self.inputs.label.output,
            self._nnet_node.inputs.label
        )

        # switching train/test is done by input
        self._constant_zero = ConstantNode([1], constant=0, name="zero")
        self._constant_one = ConstantNode([1], constant=1, name="one")
        self._switch_node = SwitchNode(2)  # outputs 1 if is_testing

        self.add_node(self._constant_zero)
        self.add_node(self._constant_one)
        self.add_node(self._switch_node)

        Connector.connect(
            self._constant_zero.outputs.output,
            self._switch_node.inputs[0]
        )
        Connector.connect(
            self._constant_one.outputs.output,
            self._switch_node.inputs[1]
        )
        Connector.connect(
            self._switch_node.outputs.output,
            self._nnet_node.inputs.testing_phase
        )

        self._is_training = True

    def _switch_training_to(self, training: bool):
        self._is_training = training
        self._switch_node.change_input(0 if training else 1)

    # support this in order to provide the data for the template
    def get_average_log_delta_for(self, layer_id: int) -> float:
        """Return 0 if not supported

        This actually returns loss during last training (average over all epochs)
        """
        return self._nnet_node.last_train_loss

    def clone_predicted_label_tensor_output(self) -> torch.Tensor:
        """Returns 1D torch.tensor of length num_labels - output of the network"""
        return self._nnet_node.outputs.label.tensor.clone().view(1, -1)

    def model_switch_to_training(self):
        """Turn learning on"""
        self._switch_training_to(True)

    def model_switch_to_testing(self):
        """Turn learning off"""
        self._switch_training_to(False)

    def model_is_learning(self):
        """This is value of the input to the NN deciding whether the TESTING phase is on"""
        return self._nnet_node.inputs.testing_phase.tensor.cpu().item() == 0

