from abc import abstractmethod
from math import ceil
from typing import List

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, Softmax

from torchsim.core.eval.metrics.abstract_classifier import AbstractClassifier
from torchsim.core.utils.tensor_utils import id_to_one_hot


class NNClassifier(AbstractClassifier):
    _trainer: "ClassifierTrainer"

    def __init__(self):
        self._trainer = None

    def _train(self, inputs: torch.Tensor, labels: torch.Tensor, n_classes: int):
        self._trainer = train_nn_classifier(inputs, labels, n_classes)

    def _evaluate(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        return self._trainer.compute_accuracy(inputs, labels)


def compute_nn_classifier_accuracy(inputs: torch.Tensor,
                                   labels: torch.Tensor,
                                   n_classes: int,
                                   custom_input_length: int = None,
                                   learn_rate: float=0.1,
                                   max_loss_difference: float =0.005,
                                   loss_window_length: int =5,
                                   max_epochs: int = 100,
                                   use_hidden_layer: bool = False,
                                   log_loss: bool = False) -> float:
    """Uses a simple (low-VC dimension) classifier to learn the labels.

    Accuracy of this classifier is returned as a measure of quality of the representation.

    Args:
        inputs: matrix of data [n_samples, input_size] - float (long shaped as [n_samples] if inputs_are_ids is True)
        labels: vector of labels [n_samples] - long
        n_classes: how many classes there is
        custom_input_length: by default, the inputs are expected to be [n_samples, input_size], if this parameter is set,
        inputs are expected to be vector of [n_samples] scalars, which is converted into one-hot format of this length
        learn_rate: 0.01, it uses SGD
        max_loss_difference: if the max difference between the loss values is smaller than this, stop training
        loss_window_length: from how long history of the loss values to determine when to stop training?
        max_epochs: in case that the data are too difficult, the training does not converge, this stops it
        use_hidden_layer: use a classifier with hidden layer (should be false)
        log_loss: should the loss be printed in time?

    Returns:
        accuracy of the classifier ~ quality of the input representation
    """
    if len(inputs.shape) != 2 and custom_input_length is None:
        raise ValueError('input tensor is expected to have 2D shape [n_samples, data_length], view it appropriately')

    if custom_input_length is not None:
        if len(inputs.shape) != 1:
            raise ValueError('in case inputs_are_ids, the shape should be 1D [n_samples]')
        inputs = id_to_one_hot(inputs, vector_len=custom_input_length)

    trainer = train_nn_classifier(inputs, labels, n_classes, learn_rate, max_loss_difference,
                                  loss_window_length, max_epochs, use_hidden_layer, log_loss)
    return trainer.compute_accuracy(inputs, labels)


def train_nn_classifier(inputs: torch.Tensor,
                        labels: torch.Tensor,
                        n_classes: int,
                        learn_rate: float = 0.1,
                        max_loss_difference: float = 0.005,
                        loss_window_length: int = 5,
                        max_epochs: int = 100,
                        use_hidden_layer: bool = False,
                        log_loss: bool = False
                        ) -> "ClassifierTrainer":
    """Trains and returns a simple neural net classifier."""
    device = str(inputs.device)
    input_size = inputs.shape[1]

    # create the network
    if use_hidden_layer:
        net = ClassifierHiddenNet(input_size=input_size, hidden_size=n_classes, n_classes=n_classes, device=device)
    else:
        net = ClassifierNoHiddenNet(input_size=input_size, n_classes=n_classes, device=device)

    trainer = ClassifierTrainer(net,
                                device=device,
                                learn_rate=learn_rate,
                                max_loss_difference=max_loss_difference,
                                loss_window_length=loss_window_length,
                                max_epochs=max_epochs,
                                log_loss=log_loss)

    trainer.train(inputs, labels)

    return trainer


class ClassifierHiddenNet(torch.nn.Module):
    """Simple classifier with one hidden layer and softmax output."""

    _input_size: int
    _output_size: int
    _hidden_size: int

    def __init__(self, input_size: int, hidden_size: int, n_classes: int, device: str):
        super(ClassifierHiddenNet, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = n_classes

        self._hidden = Linear(self._input_size, self._hidden_size)
        self._relu = ReLU()

        self._out = Linear(self._hidden_size, self._output_size)
        self._softmax = Softmax(dim=1)

        self.to(device)

    def forward(self, input):
        hidden_activations = self._hidden(input)
        hidden_activations = self._relu(hidden_activations)

        output_activations = self._out(hidden_activations)

        if self.training is False:
            output_activations = self._softmax(output_activations)

        return output_activations


class ClassifierNoHiddenNet(torch.nn.Module):
    """Simple classifier with only input/output (softmax) layer."""

    _input_size: int
    _output_size: int

    def __init__(self, input_size: int, n_classes: int, device: str):
        super(ClassifierNoHiddenNet, self).__init__()

        self._input_size = input_size
        self._output_size = n_classes

        self._out = Linear(self._input_size, self._output_size)
        self._softmax = Softmax(dim=1)

        self.to(device)

    def forward(self, input):
        output_activations = self._out(input)

        if self.training is False:
            output_activations = self._softmax(output_activations)

        return output_activations


class TrainStoppingCriterion:

    @abstractmethod
    def training_should_stop(self) -> bool:
        pass


class LossDifferenceTrainStoppingCriterion(TrainStoppingCriterion):
    """Automatically determines whether the training should be stopped.

    It is detecting the loss convergence.
    """

    _maximum_loss_difference: float
    _window_length: int
    _max_epochs: int

    _loss_history: List[float]
    _current_epoch: int

    def __init__(self, max_loss_difference: float, window_length=5, max_epochs=100):
        """The window_length of loss values is collected.

        If the max difference between the values in the history is smaller tham max_loss_difference, training should stop.

        Args:
            max_loss_difference: if abs(min(loss)-max(loss) is smaller than this value, training should stop
            window_length: how many loss values is collected
            max_epochs: maximum no of epochs the thing can be trained
        """

        self._window_length = window_length
        self._maximum_loss_difference = max_loss_difference
        self._loss_history = []
        self._max_epochs = max_epochs
        self._current_epoch = 0

    def register_loss(self, loss_val: float):
        self._loss_history.append(loss_val)
        if len(self._loss_history) > self._window_length:
            del self._loss_history[0]

    def training_should_stop(self) -> bool:

        # list not full yet?
        if len(self._loss_history) < self._window_length:
            return False

        self._current_epoch += 1

        if self._current_epoch >= self._max_epochs:
            return True

        return abs(min(self._loss_history) - max(self._loss_history)) < self._maximum_loss_difference


class BatchSampler:
    """A simple thing which randomly samples data to the batch of given samples.

    Each sample should be sampled approximately once.
    """

    _inputs: torch.Tensor
    _labels: torch.Tensor

    _batch_inputs: torch.Tensor
    _batch_labels: torch.Tensor

    _batch_size: int
    _max_batch_samples: int
    _batch_sampled_times: int

    _n_samples: int
    _data_size: int

    _device: str

    def __init__(self, inputs: Variable, labels: Variable, device: str, batch_size: int=32):

        self._device = device

        self._batch_sampled_times = 0
        self._batch_size = batch_size

        self._inputs = inputs
        self._labels = labels

        self._n_samples = inputs.shape[0]
        self._max_batch_samples = ceil(self._n_samples / self._batch_size)
        self._data_size = inputs.shape[1]

        self._batch_inputs = torch.zeros(self._inputs.shape, device=device)
        self._batch_labels = torch.zeros(self._labels.shape, device=device)

    def reset_sampler(self):
        self._batch_sampled_times = 0

    def sample_to_batch(self):

        self._batch_sampled_times += 1

        if self._batch_sampled_times > self._max_batch_samples:
            return None, None

        indexes = torch.randint(low=0, high=self._n_samples, size=[self._batch_size], device=self._device)
        idx = indexes.view(-1, 1).long()

        expanded_idnexes = idx.expand([-1, self._data_size])  # don't change the first dim, expand across the second one
        self._batch_inputs = torch.gather(input=self._inputs, dim=0, index=expanded_idnexes)

        self._batch_labels = torch.gather(input=self._labels.unsqueeze(1), dim=0, index=idx).squeeze(1)

        return Variable(self._batch_inputs), Variable(self._batch_labels)


class ClassifierTrainer:
    """A thing which trains a simple classifier until convergence.

    The accuracy of the classifier can be then used as an a measure of usefulness of the input data format (representation).
    """

    _device: str
    _learn_rate: float
    _net: torch.nn.Module
    _log_loss: bool
    _batch_size: int
    _max_epochs: int

    def __init__(self,
                 net: torch.nn.Module,
                 device: str,
                 learn_rate: float = 0.01,
                 max_loss_difference: float = 0.0005,
                 loss_window_length: int = 5,
                 max_epochs: int = 100,
                 log_loss: bool = False,
                 batch_size: int = 32):
        """This can train the network (until reasonable convergence) and compute accuracy (on the training set).

        Args:
            net: simple classifier with or without hidden layer, output is softmax
            device: cpu/gpu
            learn_rate: speed of learning through SGD
            max_loss_difference: if the max difference in the loss values is smaller than this, stop training
            loss_window_length: history length
            max_epochs: in case the learning does not converge, the normal stopping criterion does not work
            log_loss: should the loss during training be logged into the console? (debugging)
            batch_size: data are automatically split into batches (sequentially)
        """

        self._device = device
        self._learn_rate = learn_rate
        self._net = net
        self._log_loss = log_loss
        self._batch_size = batch_size
        self._max_epochs = max_epochs

        self._optimizer = torch.optim.SGD(self._net.parameters(), lr=self._learn_rate)
        self._loss_func = torch.nn.CrossEntropyLoss()

        self._stopping_criterion = LossDifferenceTrainStoppingCriterion(
            max_loss_difference,
            loss_window_length,
            max_epochs)

    def train(self, data_tensor, labels_tensor):
        """Trains the network max_train_steps on the data.

        Args:
            data_tensor: expected format [no_samples, data_dim]
            labels_tensor: expected format [no_samples, 1]

        Returns:
            Result of the last forward pass (no softmax applied during training).
        """

        data = Variable(data_tensor)
        labels = Variable(labels_tensor).squeeze()  # should be vector of target IDs

        self._net.train()
        epoch = 0

        batch_sampler = BatchSampler(data, labels, self._device)

        output = None
        # repeat epochs until the learning converged
        while True:

            batch_sampler.reset_sampler()
            batch_no = 0

            # sample the entire set into mini batches
            loss = None
            while True:
                data, labels = batch_sampler.sample_to_batch()
                batch_no += 1
                if data is None:
                    break

                output = self._net.forward(data)
                loss = self._loss_func(output, labels)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()  # apply gradients

            if self._log_loss:
                print(f'classifier loss during epoch no. {epoch} is : {loss}')

            # determine the training should stop
            self._stopping_criterion.register_loss(loss)
            if self._stopping_criterion.training_should_stop():
                break

            epoch += 1

        return output

    def compute_accuracy(self, inputs, labels):
        """Computes the accuracy of the network on the given input data and labels.

        Args:
            inputs: input data
            labels: labels belonging to the input

        Returns:
            Accuracy of the classifier.
        """

        self._net.training = False
        outputs = self._net.forward(inputs)

        arg_max_outputs = torch.max(outputs, dim=1)[1]
        comparison = arg_max_outputs == labels

        # has to be converted to float, otherwise the byte overflows at 255
        accuracy = float(comparison.sum()) / float(comparison.numel())

        return accuracy
