import copy
import os

import torch
import torch.nn as nn

from typing import List

from torchsim.core.graph.slot_container import Inputs, MemoryBlocks
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver
from torchsim.gui.observables import ObserverPropertiesItem
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import SubsetRandomSampler

from torchsim.core.graph import set_global_seeds, logging
from torchsim.core.graph.unit import Unit
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.utils.baselines_utils import parse_vars, BaselinesParams, ObservationStorage
from torchsim.utils.space_engineers_connector import SpaceEngineersConnectorConfig

logger = logging.getLogger(__name__)


class NNetParams(BaselinesParams):
    lr: float
    buffer_size: int
    batch_size: int
    num_epochs: int
    seed: int
    input_shape: tuple
    output_size: int  # TODO this is not used SEConnectorConfig instead
    curriculum: tuple
    mixed_mode: bool

    _default_params = {
        'lr': 1e-3,  # learning rate of the optimizer
        'buffer_size': 512,  # size of buffer that stores data TODO: divisibility check
        'batch_size': 128,  # size of mini-batch which is used for training
        'num_epochs': 10,  # number of weight updates per step TODO
        'seed': 1,  # seed for random number generator
        'input_shape': (3, 64, 64),  # Shape of input into the NN (default: 3x64x64)
        'output_size': 20,  # Size of the output of the NN (default: 20 for T0)
        'curriculum': (0, 1, 2, 3, -1),  # custom curriculum list e.g. (0, -1) (default: None)
        'mixed_mode': True  # whether cpu/gpu use should be mixed
    }

    def __eq__(self, other):
        if not isinstance(other, NNetParams):
            return False

        # get all attributes of this class
        attrs = vars(self)

        # compare each of them with other
        for attribute_name in attrs:
            a = getattr(self, attribute_name)
            b = getattr(other, attribute_name)

            if a != b:
                return False
        return True


class NNetNodeUnit(Unit):

    params: NNetParams
    storage: ObservationStorage
    optimizer: Optimizer
    network: Module
    output: Tensor
    label: Tensor
    task_control: Tensor
    cur_train_step: int

    last_train_loss: float

    def __init__(self, creator: TensorCreator,
                 network: nn.Module,
                 optimizer: Optimizer,
                 storage: ObservationStorage,
                 params: NNetParams):
        """Unit constructor.

        Args:
            creator: creator of this node
            network: pytorch neural network module
            optimizer: pytorch optimizer object
            storage: baselines observation storage object
            params: baselines parameter object
        """
        super().__init__(creator.device)

        self.params = params
        self.network = network
        self.optimizer = optimizer
        self.storage = storage
        self.device = creator.device

        self.output = creator.zeros(
            SpaceEngineersConnectorConfig().task_to_agent_buffer_size,
            dtype=self._float_dtype, device=self._device)
        self.label = creator.zeros(
            SpaceEngineersConnectorConfig().task_to_agent_buffer_size,
            dtype=self._float_dtype, device=self._device)
        self.task_control = creator.zeros(
            4, device=self._device)

        self.cur_train_step = 0
        self.last_train_loss = 0

    def _save(self, saver: Saver):
        super()._save(saver)

        # TODO the storage should be serialzied/deserialized here ?
        # saver.description['store_x'] = self.storage.x

        torch.save(self.network, os.path.join(saver.get_full_folder_path(), 'network.pt'))

    def _load(self, loader: Loader):
        super()._load(loader)

        # TODO remove this after deserialization fixed
        logger.error("nn_node.py: loading contains some bug which breaks the learning, " +
                     "please manually disable loading in the TestableExperimentTemplateBase " +
                     "(comment out the line 477: self._topology_saver.load_data_into(self._topology) )")

        # self.storage.x = loader.description['store_x']
        self.network = torch.load(os.path.join(loader.get_full_folder_path(), 'network.pt'))

    def update(self):
        """Performs update of model parameters.

        Unlike in standard learning settings, model weights are updated
        over a (predefined) number of epochs. In each epoch a batch generator
        is created from which batches of data are sampled and optimizer updates
        weights of a model based on said batch.
        """
        if self.params.mixed_mode:
            self.storage.to('cuda')

        # run update for a number of epochs
        for e in range(self.params.num_epochs):

            # sample data from buffer
            data_generator = self.storage.generator(
                self.params.batch_size,
                sampler=SubsetRandomSampler,
                indices=range(self.params.buffer_size))

            epoch_loss: int = 0

            # update weights for each sample batch
            for batch in data_generator:
                self.optimizer.zero_grad()
                output = self.network(batch[0])
                target = batch[1]  # TODO: make access neater/clearer
                loss = self.network.criterion(output, target.argmax(1))
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step(None)

            logger.info(f'current epoch loss: {epoch_loss/self.params.num_epochs}')
            self.last_train_loss = epoch_loss/self.params.num_epochs

    def step(self, x: torch.Tensor, y: torch.Tensor, is_testing: Tensor):
        """Simulator step.

        Args:
            x: observations
            y: targets
            is_testing: flag for training/testing stage
        """

        # only performed during training
        if not is_testing.item():
            # store current observation
            self.storage.insert({
                'x': x,
                'y': y
            })

            # update weights once enough data is collected
            if self.cur_train_step % self.params.buffer_size == 0 and self.cur_train_step > 0:
                self.update()
                self.storage.after_update({'x', 'y'})
                if self.params.mixed_mode:
                    self.storage.to('cpu')

            self.cur_train_step += 1

        # make prediction
        yhat = self.network(x.unsqueeze(0).to(
            'cuda' if self.params.mixed_mode else self.device))

        # return predictions
        self.output.zero_()
        self.label.zero_()
        self.output[yhat.argmax().item()] = 1
        self.label[yhat.argmax().item()] = 1


class NNetNodeInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create("Input")
        self.label = self.create("Label")
        self.testing_phase = self.create("Testing_Phase")


class NNetNodeOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")
        # required outputs
        self.label = self.create("Label")
        self.task_control = self.create("Task Control")

    def prepare_slots(self, unit: NNetNodeUnit):
        self.output.tensor = unit.output
        # required
        self.label.tensor = unit.label
        self.task_control.tensor = unit.task_control


class NNetNode(WorkerNodeBase[NNetNodeInputs, NNetNodeOutputs]):
    """Neural Network Node."""

    _unit: NNetNodeUnit

    def __init__(self,
                 network: nn.Module,
                 optimizer: Optimizer,
                 storage: ObservationStorage,
                 params: NNetParams,
                 name=""):
        super().__init__(
            name=name,
            inputs=NNetNodeInputs(self),
            outputs=NNetNodeOutputs(self))

        self._network = network
        self._optimizer = optimizer
        self._storage = storage
        self._params = copy.copy(params)

    def _create_unit(self, creator: TensorCreator):
        set_global_seeds(self._params.seed)
        return NNetNodeUnit(creator,
                            self._network, self._optimizer,
                            self._storage, self._params)

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return parse_vars(self._params)

    def _step(self):
        self._unit.step(
            self.inputs.input.tensor.permute(2, 0, 1),  # NOTE: permutation due to SE
            self.inputs.label.tensor,
            self.inputs.testing_phase.tensor
        )

    @property
    def last_train_loss(self):
        return self._unit.last_train_loss

