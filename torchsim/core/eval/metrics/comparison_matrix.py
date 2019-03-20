from typing import List, Callable, Union

import torch


def comparison_matrix(series: List[torch.LongTensor],
                      comparison: Callable[[torch.LongTensor, torch.LongTensor], float],
                      empty_element_value: float = -1,
                      self_comparison_value: float = 1
                      ) -> torch.Tensor:
    """Computes an upper triangular matrix of a similarity measure.

    Args:
        series: A list of 1D tensors of equal length, each representing a series of ids
        comparison: Function comparing two tensors, returning a similarity value
        empty_element_value: Value filling the lower part of the matrix
        self_comparison_value: Value representing comparison of identical items
    Returns:
        A 2D tensor with a triangular matrix of comparisons
    """
    size = len(series)
    comparisons = torch.full([size, size], empty_element_value)
    for i in range(size):
        comparisons[i, i] = self_comparison_value
        for j in range(i + 1, size):
            comparisons[i, j] = comparison(series[i], series[j])
    return comparisons


def classification_matrix(classifier_class: type,
                          inputs: Union[List[List[int]], List[torch.Tensor], List[List[torch.Tensor]]],
                          labels: Union[List[int], torch.Tensor],
                          n_classes: int,
                          classifier_input_size: int = None,
                          device: str = 'cpu',
                          empty_element_value: float = -1
                          ) -> torch.Tensor:
    """Computes an upper triangular matrix of classification accuracies.

    Args:
        classifier_class: The class of the classifier to instantiate, a subclass of AbstractClassifier
        inputs: The training data (list of: list or 1D tensor of integer ids or 2D tensor or list of tensor)
        labels: The ground truth labels (integers)
        n_classes: The number of classes
        classifier_input_size: Size of one-hot vectors; must be specified when inputs are integer ids
        device: Device of tensors; must be specified when inputs are lists
        empty_element_value: Value filling the lower part of the matrix

    Returns:
        A 2D tensor with a triangular matrix of classification accuracies
    """
    size = len(inputs)
    accuracies = torch.full([size, size], empty_element_value)
    for i in range(size):
        classifier = classifier_class()
        accuracies[i, i] = classifier.train_and_evaluate(inputs[i], labels, n_classes=n_classes,
                                                         classifier_input_size=classifier_input_size, device=device)
        for j in range(i + 1, size):
            accuracies[i, j] = classifier.evaluate(inputs[j], labels, classifier_input_size=classifier_input_size,
                                                   device=device)
    return accuracies
