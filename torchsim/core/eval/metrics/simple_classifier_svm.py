import numpy as np
from sklearn.svm import SVC

import torch
from sklearn import svm
from typing import List, Optional

from torchsim.core.eval.metrics.abstract_classifier import AbstractClassifier
from torchsim.core.utils.tensor_utils import id_to_one_hot, one_hot_to_id


class SvmClassifier(AbstractClassifier):
    _model: "SvmModel"

    def _train(self, inputs: torch.Tensor, labels: torch.Tensor, n_classes: int):
        self._model = train_svm_classifier(inputs, labels, n_classes)

    def _evaluate(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        return evaluate_svm_classifier(self._model, inputs, labels)


def compute_svm_classifier_accuracy(inputs: torch.Tensor,
                                    labels: torch.Tensor,
                                    n_classes: int,
                                    custom_input_length: int = None
                                    ) -> float:
    """Trains the SVM classifier on the data and returns its classification accuracy on the same data.

    Similar to low-VC dimensional classifier in the compute_nn_classifier_accuracy method,
    this might train faster and might be less prone to non-determinism introduced by the training of NN.

    Args:
        inputs: matrix of data [n_samples, input_size] or long shaped as [n_samples] if custom_input_length is set
        labels: vector of labels [n_samples] or one-hot vector [n_samples, n_classes]
        n_classes: how many classes there is
        custom_input_length: if set to value, inputs is interpreted as ids and converted to one-hot vector

    Returns:
        Accuracy of the classifier ~ quality of the input representation.
    """
    model = train_svm_classifier(inputs, labels, n_classes, custom_input_length)
    return evaluate_svm_classifier(model, inputs, labels)


def train_svm_classifier(inputs: torch.Tensor,
                         labels: torch.Tensor,
                         n_classes: int,
                         custom_input_length: int = None
                         ) -> "SvmModel":
    model = SvmModel(num_classes=n_classes)

    if len(inputs.shape) != 2 and custom_input_length is None:
        raise ValueError('input tensor is expected to have 2D shape [n_samples, data_length], view it appropriately')

    if custom_input_length is not None:
        inputs = id_to_one_hot(inputs, custom_input_length)

    if len(labels.shape) >= 2:
        labels = one_hot_to_id(labels)

    inputs = np.array(inputs.cpu())
    labels = np.array(labels.cpu())

    model.train(inputs, labels)
    return model


def evaluate_svm_classifier(model: "SvmModel",
                            inputs: torch.Tensor,
                            labels: torch.Tensor) -> float:
    return (torch.tensor(model.predict(inputs.cpu().numpy())) ==
            labels.cpu()).type(torch.float).mean().item()


class SvmParamSearchParams:
    split: float = 0.75
    runs: int = 1
    learn_class_balancing: bool = False
    class_balancing_trials: int = 1000
    fixed_class_balancing: List[float] = None


class SvmParams:
    param_search: SvmParamSearchParams
    kernel: str = 'linear'
    gamma: str = 'auto'
    c: float = 1.


class SvmModel:
    """Svm model with inner loop for param search."""
    num_classes: int
    _trained_model: Optional[SVC]

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def train(self, samples: np.ndarray, targets: np.ndarray, params: SvmParams = SvmParams()):
        self._trained_model = self._train_model(samples, targets, kernel=params.kernel, c=params.c, gamma=params.gamma)

    def predict(self, testing_data: np.ndarray) -> np.ndarray:
        """Compute prediction probabilities weighted by learned balancing."""
        if self._trained_model is None:
            raise RuntimeError('Model has not been trained')

        predictions = self._trained_model.predict(testing_data)

        return predictions

    @staticmethod
    def _train_model(samples, targets, kernel, c, gamma) -> SVC:
        if np.all(targets[0] == targets):
            const = targets[0]

            class SingleTargetSVCPredictor(SVC):
                """Simple SVC predictor that works for just one category

                Note: just predict() gives correct output
                """
                def predict(self, data: np.ndarray) -> np.ndarray:
                    return np.full((data.shape[0],), const)

            return SingleTargetSVCPredictor()

        model = svm.SVC(C=c, kernel=kernel, gamma=gamma)
        model.fit(samples, targets)

        return model
