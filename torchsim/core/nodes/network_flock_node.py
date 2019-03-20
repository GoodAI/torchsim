from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import torch
from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.slot_container_base import GenericMemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeWithInternalsBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.core.models.neural_network.network_flock_buffer import NetworkFlockBuffer
from torchsim.core.nodes.flock_networks.delay_buffer import DelayBuffer, create_delay_buffer
from torchsim.core.nodes.flock_networks.network_factory import create_networks
from torchsim.core.nodes.flock_networks.neural_network_flock import NeuralNetworkFlock, NeuralNetworkFlockTypes, \
    NeuralNetworkFlockParams
from torchsim.gui.observables import ObserverPropertiesItem, enable_on_runtime, disable_on_runtime
from torchsim.gui.validators import validate_positive_int, validate_predicate, validate_float_in_range, \
    validate_positive_with_zero_int
from torch import FloatTensor


@dataclass
class NetworkFlockNodeParams(ParamsBase):
    flock_size: int = 1
    buffer_size: int = 300
    batch_size: int = 200

    is_learning_enabled: bool = True
    learning_period: int = 10

    do_delay_input: bool = True  # delay the data in order to sync with other inputs (e.g. targets)?
    do_delay_coefficients: bool = False  # delay the coefficients as well?

    normalize_coefficients: bool = False
    negative_coefficients_removal: bool = False
    network_type: NeuralNetworkFlockTypes = NeuralNetworkFlockTypes.MLP  # type of the network to be used


class LearningEvaluator:
    """Determines whether the learning should happen at this time step"""

    _node_params: NetworkFlockNodeParams
    _network_params: NeuralNetworkFlockParams
    _step: int

    def __init__(self, params: NetworkFlockNodeParams, network_params: NeuralNetworkFlockParams):
        self._node_params = params
        self._network_params = network_params

        self._step = 0

    def should_learn(self, buffer: NetworkFlockBuffer) -> bool:
        """batch can be sampled now, learning is enabled, learning rate is > 0 and learning period is fulfilled"""

        self._step += 1

        return (
                buffer.can_sample_batch(self._node_params.batch_size) and
                self._node_params.is_learning_enabled and
                self._network_params.learning_rate > 0.0 and
                self._step % self._node_params.learning_period == 0
        )


class NetworkFlockUnit(Unit):
    _creator: TensorCreator
    _params: NetworkFlockNodeParams
    _network_params: NeuralNetworkFlockParams
    _input_shape: Tuple
    _target_shape: Tuple

    _learning_evaluator: LearningEvaluator

    buffer: NetworkFlockBuffer

    _learning_coefficients_delay: DelayBuffer
    _input_data_delay: DelayBuffer

    last_prediction_output: torch.Tensor
    last_error_output: torch.Tensor

    _networks: NeuralNetworkFlock

    def __init__(self,
                 creator: TensorCreator,
                 params: NetworkFlockNodeParams,
                 network_params: NeuralNetworkFlockParams,
                 input_shape: Tuple,
                 target_shape: Tuple,
                 networks: NeuralNetworkFlock):
        """
        Creates an instance of the unit.

        Args:
            creator:
            params:
            input_shape: expected shape of the input is without flock_size (the input is expanded flock_size times)
            target_shape: expected shape of the target is without the flock_size as well
            (output size is flock_size, *target_shape)
            networks: an instance of NeuralNetworkFlock, which holds flock_size of identical networks
        """
        super().__init__(creator.device)

        self._params = params
        self._network_params = network_params
        self._input_shape = input_shape
        self._target_shape = target_shape
        self._creator = creator
        self._networks = networks

        self.buffer = NetworkFlockBuffer(
            creator=creator,
            flock_size=self._params.flock_size,
            buffer_size=self._params.buffer_size,
            input_shape=self._input_shape,
            target_shape=self._target_shape,
            delay_used=self._params.do_delay_input or self._params.do_delay_coefficients
        )

        # sampled batch with inputs
        self.inputs_batch = creator.zeros((self._params.flock_size,
                                           self._params.batch_size,
                                           *self._input_shape),
                                          device=creator.device,
                                          dtype=creator.float)
        # sampled batch with targets
        self.targets_batch = creator.zeros((self._params.flock_size,
                                            self._params.batch_size,
                                            *self._target_shape),
                                           device=creator.device,
                                           dtype=creator.float)

        self.coefficients_batch = creator.zeros((self._params.flock_size,
                                                 self._params.batch_size,
                                                 1),
                                                device=creator.device,
                                                dtype=creator.float)

        # backup some inputs in order to synchronize data written to the buffers (data, targets, weights)
        self._input_data_delay = create_delay_buffer(creator,
                                                     self._params.do_delay_input,
                                                     (self._params.flock_size, *self._input_shape))
        self._learning_coefficients_delay = create_delay_buffer(creator,
                                                                self._params.do_delay_coefficients,
                                                                (self._params.flock_size, 1))

        # outputs of the node
        self.last_prediction_output = creator.zeros((self._params.flock_size, *target_shape),
                                                    dtype=creator.float,
                                                    device=creator.device)

        self.last_error_output = creator.zeros((self._params.flock_size, 1),
                                               dtype=creator.float,
                                               device=creator.device)

        self._learning_evaluator = LearningEvaluator(self._params, self._network_params)

        self.is_learning = creator.zeros((self._params.flock_size,), dtype=creator.float, device=creator.device)

    def _resize_input(self, input: FloatTensor) -> torch.Tensor:
        """Resize the input [*input_dims] so that it is compatible with the [flock_size, *input_dims] format"""
        return input.unsqueeze(0).expand((self._params.flock_size,) + input.shape)

    def step(self, input_data: FloatTensor, target_data: FloatTensor, learning_coefficients: FloatTensor):
        # reshape inputs for the compatibility
        flock_input_data = self._resize_input(input_data)
        flock_target_data = self._resize_input(target_data)

        if self._params.negative_coefficients_removal:
            if learning_coefficients.min().item() < 0.:
                learning_coefficients += -learning_coefficients.min()

        if self._params.normalize_coefficients:
            if learning_coefficients.max().item() != 0.:
                learning_coefficients /= learning_coefficients.max()

        flock_coefficients = learning_coefficients.unsqueeze(-1)

        # learn the networks if possible
        if self._learning_evaluator.should_learn(self.buffer):
            self.buffer.sample_learning_batch(self._params.batch_size,
                                              self.inputs_batch,
                                              self.targets_batch,
                                              self.coefficients_batch)

            # create the data loaders containing correctly shaped inputs/outputs and coefficients
            learning_size = (self._params.flock_size, self._params.batch_size, -1)

            loaders, should_learn = self._networks.make_data_loaders(
                self.inputs_batch.view(learning_size),
                self.targets_batch.view(learning_size),
                self.coefficients_batch.view(learning_size),
                self._network_params.mini_batch_size,
                self._network_params.coefficients_minimum_max
            )

            self.is_learning.copy_(torch.tensor(should_learn, dtype=torch.float))

            # update the network
            # note: if all the coefficients are zeros, the loaders throw some exception
            self._networks.train(loaders, should_learn)

        # zero the is_learning vector
        if not self._params.is_learning_enabled:
            self.is_learning.zero_()

        # compute the prediction error (has to be here, before the last_prediction_output is rewritten
        errors = self._networks.compute_errors(self.last_prediction_output, flock_target_data)
        # we need to add the singleton dimension because last_error_output has shale [flock_size, 1]
        self.last_error_output.copy_(errors.unsqueeze(dim=1))

        # run the forward pass(es)
        # unsqueeze to add a dimension for batch - forward_pass computes with batches
        self._networks.forward_pass(flock_input_data.unsqueeze(1),
                                    self.last_prediction_output)

        if self._params.is_learning_enabled:
            # align the IO delays
            self._input_data_delay.push(flock_input_data)
            self._learning_coefficients_delay.push(flock_coefficients)
            # store data to the buffer for learning
            self.buffer.store(self._input_data_delay.read(),
                              flock_target_data,
                              self._learning_coefficients_delay.read())


class NetworkFlockInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input_data = self.create("Input data")
        self.target_data = self.create("Target data")
        self.learning_coefficients = self.create("Learning coefficients")


class NetworkFlockInternals(GenericMemoryBlocks['NetworkFlockNode']):
    """Class which holds the internals for a NetworkFlockNode.

    Args:
        owner (NetworkFlockNode): The node to which these internals belong to.
    """

    def __init__(self, owner: 'NetworkFlockNode'):
        super().__init__(owner)

        self.buffer_total_data_written = self.create("Buffer_total_data_written")

        self.buffer_inputs = self.create_buffer("Buffer_inputs")
        self.buffer_targets = self.create_buffer("Buffer_targets")
        self.buffer_coefficients = self.create_buffer("Buffer_coefficients")

        self.batch_inputs = self.create("Batch_inputs")
        self.batch_targets = self.create("Batch_targets")
        self.batch_coefficients = self.create("Batch_coefficients")

        self.is_learning = self.create("Is learning")

    def prepare_slots(self, unit: NetworkFlockUnit):
        self.buffer_total_data_written.tensor = unit.buffer.total_data_written

        self.buffer_inputs.buffer = unit.buffer.inputs
        self.buffer_targets.buffer = unit.buffer.targets
        self.buffer_coefficients.buffer = unit.buffer.learning_coefficients

        self.batch_inputs.tensor = unit.inputs_batch
        self.batch_targets.tensor = unit.targets_batch
        self.batch_coefficients.tensor = unit.coefficients_batch

        self.is_learning.tensor = unit.is_learning


class NetworkFlockOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.prediction_output = self.create("Prediction output")
        self.error_output = self.create("Error output")

    def prepare_slots(self, unit: NetworkFlockUnit):
        self.prediction_output.tensor = unit.last_prediction_output
        self.error_output.tensor = unit.last_error_output


class NetworkFlockNode(WorkerNodeWithInternalsBase[NetworkFlockInputs, NetworkFlockInternals, NetworkFlockOutputs]):
    """A simple node which has a buffer
    """

    _unit: NetworkFlockUnit
    _node_params: NetworkFlockNodeParams
    _network_params: NeuralNetworkFlockParams
    _input_shape: Tuple
    _target_shape: Tuple

    def __init__(self,
                 name="NetworkFlockNode",
                 node_params: Optional[NetworkFlockNodeParams] = None,
                 network_params: Optional[NeuralNetworkFlockParams] = None):
        super().__init__(name=name,
                         inputs=NetworkFlockInputs(self),
                         memory_blocks=NetworkFlockInternals(self),
                         outputs=NetworkFlockOutputs(self))

        self._node_params = node_params.clone() if node_params else NetworkFlockNodeParams()
        self._network_params = network_params.clone() if network_params else NeuralNetworkFlockParams()

    def _create_unit(self, creator: TensorCreator) -> NetworkFlockUnit:
        self._derive_params()

        networks = create_networks(network_params=self._network_params,
                                   creator=creator,
                                   network_type=self._node_params.network_type)

        return NetworkFlockUnit(creator,
                                params=self._node_params,
                                network_params=self._network_params,
                                input_shape=self._input_shape,
                                target_shape=self._target_shape,
                                networks=networks)

    def _step(self):
        self._unit.step(self.inputs.input_data.tensor,
                        self.inputs.target_data.tensor,
                        self.inputs.learning_coefficients.tensor)

    def _derive_params(self):
        """Derive the params of the node from the input shape."""
        self._input_shape = tuple(self.inputs.input_data.tensor.shape)
        self._target_shape = tuple(self.inputs.target_data.tensor.shape)

        self._network_params.input_size = int(np.prod(self._input_shape))
        self._network_params.output_size = int(np.prod(self._target_shape))
        self._network_params.flock_size = self._node_params.flock_size

    def validate(self):
        validate_positive_int(self._node_params.flock_size)
        validate_predicate(lambda: self.inputs.learning_coefficients.tensor.dim() == 1)
        validate_predicate(lambda: self._node_params.flock_size == self.inputs.learning_coefficients.tensor.shape[0])

    @property
    def is_learning_enabled(self) -> bool:
        return self._node_params.is_learning_enabled

    @is_learning_enabled.setter
    def is_learning_enabled(self, value: bool):
        self._node_params.is_learning_enabled = value

    @property
    def do_delay_input(self) -> bool:
        return self._node_params.do_delay_input

    @do_delay_input.setter
    def do_delay_input(self, value: bool):
        self._node_params.do_delay_input = value

    @property
    def do_delay_coefficients(self) -> bool:
        return self._node_params.do_delay_coefficients

    @do_delay_coefficients.setter
    def do_delay_coefficients(self, value: bool):
        self._node_params.do_delay_coefficients = value

    @property
    def flock_size(self) -> int:
        return self._node_params.flock_size

    @flock_size.setter
    def flock_size(self, value: int):
        validate_positive_int(value)
        self._node_params.flock_size = value
        self._network_params.flock_size = value

    @property
    def buffer_size(self) -> int:
        return self._node_params.buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int):
        validate_positive_int(value)
        validate_predicate(lambda: value > self._node_params.batch_size)
        self._node_params.buffer_size = value

    @property
    def batch_size(self) -> int:
        return self._node_params.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        validate_positive_int(value)
        validate_predicate(lambda: value < self._node_params.buffer_size)
        self._node_params.batch_size = value

    @property
    def learning_rate(self) -> float:
        return self._network_params.learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        validate_float_in_range(value, min_range=0, max_range=1)
        self._network_params.learning_rate = value
        if self._unit is not None:
            self._unit._networks.set_learning_rate(value)

    @property
    def learning_period(self) -> int:
        return self._node_params.learning_period

    @learning_period.setter
    def learning_period(self, value: int):
        validate_positive_int(value)
        self._node_params.learning_period = value

    @property
    def hidden_size(self) -> int:
        return self._network_params.hidden_size

    @hidden_size.setter
    def hidden_size(self, value: int):
        validate_positive_int(value)
        self._network_params.hidden_size = value

    @property
    def n_hidden_layers(self) -> int:
        return self._network_params.n_hidden_layers

    @n_hidden_layers.setter
    def n_hidden_layers(self, value: int):
        validate_positive_with_zero_int(value)
        self._network_params.n_hidden_layers = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        """Define which properties can be changed from GUI and how."""
        return [
            # constant params
            self._prop_builder.auto('Flock size', type(self).flock_size,
                                    edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Buffer size', type(self).buffer_size,
                                    edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Batch size', type(self).batch_size,
                                    edit_strategy=disable_on_runtime),
            self._prop_builder.auto('N hidden layers', type(self).n_hidden_layers,
                                    edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Hidden size', type(self).hidden_size,
                                    edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Do delay input', type(self).do_delay_input,
                                    edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Do delay coefficients', type(self).do_delay_coefficients,
                                    edit_strategy=disable_on_runtime),

            # changeable during runtime
            self._prop_builder.auto('Enable learning', type(self).is_learning_enabled,
                                    edit_strategy=enable_on_runtime),
            self._prop_builder.auto('Learning period', type(self).learning_period,
                                    edit_strategy=enable_on_runtime),
            self._prop_builder.auto('Learning rate', type(self).learning_rate,
                                    edit_strategy=enable_on_runtime),
        ]
