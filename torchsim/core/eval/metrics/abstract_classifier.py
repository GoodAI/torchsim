import logging
from abc import ABC, abstractmethod
from typing import List, Union

import torch

from torchsim.core.utils.tensor_utils import id_to_one_hot

logger = logging.getLogger(__name__)


class AbstractClassifier(ABC):
    """Provides a common interface for weak classifiers used for evaluation.

    To make a concrete subclass for a specific classifier, override the methods _train and _evaluate.
    """

    def train(self,
              inputs: Union[torch.Tensor, List[int], List[torch.Tensor]],
              labels: Union[torch.Tensor, List[int]],
              n_classes: int,
              classifier_input_size: int = None,
              device: str = 'cpu'):
        """Trains the classifier with the labels and input data.

        Input data can be specified in a number of ways:
            - List[int] (need to set classifier_input_size, optionally set device)
            - 1D torch.Tensor with scalars (need to set classifier input size)
            - 2D torch.Tensor with input data
            - List of torch.Tensor

        Labels are given as
            - List[int] (optionally set device)
            - 1D tensor of integers

        The classifier_input_size and device parameters are used when converting data to the canonical
        internal format. See above for when they are needed.

        Args:
            inputs: The training data (list or 1D tensor of integer ids or 2D tensor or list of tensor)
            labels: The ground truth labels (integers)
            n_classes: The number of classes
            classifier_input_size: Size of one-hot vectors; must be specified when input is integer ids
            device: Device of tensors; must be specified when input is list

        Returns:
            Nothing
        """
        inputs = self._check_and_standardize(inputs, classifier_input_size=classifier_input_size, device=device,
                                             verify_input_tensor=True)
        labels = self._check_and_standardize(labels, device=device)
        self._train(inputs, labels, n_classes)

    @abstractmethod
    def _train(self, inputs: torch.Tensor, labels: torch.Tensor, n_classes: int):
        """Trains the classifier -- overridden in concrete subclasses.

        Args:
            inputs: Input data as 2D tensor
            labels: Labels as 1D tensor of integers
            n_classes: Number of classes

        Returns:
            Nothing
        """
        pass

    def evaluate(self,
                 inputs: Union[torch.Tensor, List[int], List[torch.Tensor]],
                 labels: Union[torch.Tensor, List[int]],
                 classifier_input_size: int = None,
                 device: str = 'cpu') -> float:
        """Evaluates the classifier on the given inputs and labels.

        Input data gets converted to a canonical internal format, just as ni the train method.

        Args:
            inputs: The training data (list or 1D tensor of integer ids or 2D tensor or list of tensor)
            labels: The ground truth labels (integers)
            classifier_input_size: Size of one-hot vectors; must be specified when input is integer ids
            device: Device of tensors; must be specified when input is list

        Returns:
            Classification accuracy in [0, 1]
        """
        inputs = self._check_and_standardize(inputs, classifier_input_size=classifier_input_size, device=device,
                                             verify_input_tensor=True)
        labels = self._check_and_standardize(labels, device=device)
        return self._evaluate(inputs, labels)

    @abstractmethod
    def _evaluate(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Evaluates the classifier -- overridden in concrete subclasses.

        Args:
            inputs: Input data as 2D tensor
            labels: Labels as 1D tensor of integers

        Returns:
            Classification accuracy in [0, 1]
        """
        pass

    def train_and_evaluate(self,
                           inputs: Union[torch.Tensor, List[int], List[torch.Tensor]],
                           labels: Union[torch.Tensor, List[int]],
                           n_classes: int,
                           classifier_input_size: int = None,
                           device: str = 'cpu'
                           ) -> float:
        """Combines training and evaluation (on the same data).

        Args:
            inputs: The training data (list or 1D tensor of integer ids or 2D tensor or list of tensor)
            labels: The ground truth labels (integers)
            n_classes: The number of classes
            classifier_input_size: Size of one-hot vectors; must be specified when input is integer ids
            device: Device of tensors; must be specified when input is list

        Returns:
            Classification accuracy in [0, 1]
        """
        self.train(inputs, labels, n_classes, classifier_input_size, device)
        return self.evaluate(inputs, labels, classifier_input_size, device)

    def train_and_evaluate_in_phases(self,
                                     inputs: Union[List[List[int]], List[torch.Tensor], List[List[torch.Tensor]]],
                                     labels: Union[List[List[int]], List[torch.Tensor], List[List[torch.Tensor]]],
                                     n_classes: int,
                                     classifier_input_size: int = None,
                                     device: str = 'cpu') -> List[float]:
        """Trains and evaluates multiple sets of inputs and labels, corresponding to testing or training phases.

        Args:
            inputs: Phase input data (each list element contains the input data for a phase)
            labels: Phase label data (each list element contains the labels for a phase)
            n_classes: The number of classes
            classifier_input_size: Size of one-hot vectors; must be specified when inputs are integer ids
            device: Device of tensors; must be specified when inputs are lists

        Returns:
            List of phase classification accuracies in [0, 1]
        """
        results = []
        phase_id = 0
        max_phases = len(inputs)

        for phase_inputs, phase_labels in zip(inputs, labels):

            logger.info(f'train and evaluating classifier in phase {phase_id} of {max_phases}')
            phase_id += 1

            acc = self.train_and_evaluate(phase_inputs,
                                          phase_labels,
                                          n_classes=n_classes,
                                          classifier_input_size=classifier_input_size,
                                          device=device)
            results.append(acc)
        return results

    def train_and_evaluate_in_phases_train_test(self,
                                                inputs: Union[
                                                    List[List[int]], List[torch.Tensor], List[List[torch.Tensor]]],
                                                labels: Union[
                                                    List[List[int]], List[torch.Tensor], List[List[torch.Tensor]]],
                                                inputs_test: Union[
                                                    List[List[int]], List[torch.Tensor], List[List[torch.Tensor]]],
                                                labels_test: Union[
                                                    List[List[int]], List[torch.Tensor], List[List[torch.Tensor]]],
                                                n_classes: int,
                                                classifier_input_size: int = None,
                                                device: str = 'cpu') -> [List[float], List[float]]:
        """ The same as train_and_evaluate_in_phases, but this one trains on the train data and evaluates both
        on train and test data.

        Args:
            inputs: Phase input data (each list element contains the input data for a phase)
            labels: Phase label data (each list element contains the labels for a phase)
            inputs_test: Phase input data from the testing phase
            labels_test: Phase label data from the testing phase
            n_classes: The number of classes
            classifier_input_size: Size of one-hot vectors; must be specified when inputs are integer ids
            device: Device of tensors; must be specified when inputs are lists

        Returns:
            List of phase classification accuracies in [0, 1]
        """
        results_train = []
        results_test = []
        phase_counter = 0
        num_phases = len(inputs)
        for phase_inputs, phase_labels, phase_test_inputs, phase_test_labels in zip(inputs, labels, inputs_test,
                                                                                    labels_test):

            logger.info(f'training classifier, phase {phase_counter} of {num_phases}')

            self.train(phase_inputs,
                       phase_labels,
                       n_classes=n_classes,
                       classifier_input_size=classifier_input_size,
                       device=device)

            logger.info(f'evaluating classifier, phase {phase_counter} of {num_phases} (train)')

            acc_train = self.evaluate(phase_inputs,
                                      phase_labels,
                                      classifier_input_size=classifier_input_size,
                                      device=device)

            logger.info(f'evaluating classifier, phase {phase_counter} of {num_phases} (test)')

            acc_test = self.evaluate(phase_test_inputs,
                                     phase_test_labels,
                                     classifier_input_size=classifier_input_size,
                                     device=device)

            results_train.append(acc_train)
            results_test.append(acc_test)
            phase_counter += 1

        logger.info(f'evaluation of the classifier done, best train: {max(results_train)}%,'
                    f' best test: {max(results_test)}%')
        return results_train, results_test

    @staticmethod
    def _check_and_standardize(data: Union[torch.Tensor, List[int], List[torch.Tensor]],
                               classifier_input_size: int = None,
                               device: str = 'cpu',
                               verify_input_tensor: bool = False) -> torch.Tensor:
        """Converts input data or labels into the format used internally.

        Args:
            data: The data (input data or labels) to be standardized
            classifier_input_size: Size of one-hot vectors; must be specified when input is integer ids
            device: Device of tensors; must be specified when input is list
            verify_input_tensor: True if data is input data (not labels)

        Returns:
            The data in a standard format
        """
        if type(data) is list and type(data[0]) is int:
            data = torch.tensor(data, device=device, dtype=torch.long)

        if type(data) is list and type(data[0]) is torch.Tensor:
            data = torch.stack(data)

        if verify_input_tensor and data.dim() != 2 and classifier_input_size is None:
            raise ValueError(
                'input tensor is expected to have 2D shape [n_samples, data_length], view it appropriately')

        if classifier_input_size is not None:
            if data.dim() != 1:
                raise ValueError('in case classifier_input_size is not None, the shape should be 1D [n_samples]')
            data = id_to_one_hot(data, vector_len=classifier_input_size)

        if torch.isnan(data).any():
            logger.error(f'data is containing NaNs')

        return data
