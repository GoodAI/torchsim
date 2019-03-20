import torch

from torchsim.core.eval.metrics.cluster_agreement import cluster_agreement
from torchsim.core.eval.metrics.comparison_matrix import comparison_matrix, classification_matrix
from torchsim.core.eval.metrics.simple_classifier_svm import SvmClassifier


def test_comparison_matrix():
    measures = [torch.LongTensor(l) for l in [[1, 1, 1, 1], [2, 1, 2, 1], [2, 1, 3, 4]]]
    n_measures = len(measures)

    matrix = comparison_matrix(measures, cluster_agreement)

    # Diagonal values are 1
    for i in range(n_measures):
        assert matrix[i, i] == 1

    # Values at i, j, i < j, are the comparison results
    assert matrix[0, 1] == 1/2
    assert matrix[0, 2] == 1/4
    assert matrix[1, 2] == 1/2

    # Values below the diagonal are -1
    for i in range(n_measures):
        for j in range(i):
            assert matrix[i, j] == -1


def test_classification_matrix():
    ground_truth = torch.LongTensor([0, 1, 2])
    inputs = [[torch.LongTensor(l) for l in [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
              [torch.LongTensor(l) for l in [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]],
              [torch.LongTensor(l) for l in [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]]]
    n_outputs = len(inputs)

    matrix = classification_matrix(SvmClassifier, inputs, ground_truth, n_classes=4)

    # Diagonal values
    for i in range(n_outputs):
        assert 0 < matrix[i, i] <= 1

    # Values above the diagonal
    for i in range(n_outputs):
        for j in range(i + 1, n_outputs):
            assert 0 <= matrix[i, j] <= matrix[i, i]

    # Values below the diagonal are -1
    for i in range(n_outputs):
        for j in range(i):
            assert matrix[i, j] == -1

