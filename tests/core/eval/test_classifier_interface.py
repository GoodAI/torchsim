import pytest
import torch

from torchsim.core.eval.metrics.simple_classifier_nn import NNClassifier
from torchsim.core.eval.metrics.simple_classifier_svm import SvmClassifier


@pytest.mark.parametrize('classifier_class', [SvmClassifier, NNClassifier])
def test_classifier_interface(classifier_class: type):
    ground_truth = torch.LongTensor([0, 1, 2])
    training_data = torch.Tensor([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    testing_data = [torch.Tensor([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
                    torch.Tensor([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
                    ]

    classifier = classifier_class()
    classifier.train(training_data, ground_truth, n_classes=3)
    accuracy_on_training_data = classifier.evaluate(training_data, ground_truth)
    assert accuracy_on_training_data == 1
    for i in range(2):
        assert 0 < classifier.evaluate(testing_data[i], ground_truth) <= accuracy_on_training_data


@pytest.mark.parametrize('classifier_class', [SvmClassifier, NNClassifier])
def test_input_type_ids(classifier_class: type):
    ground_truth = [0, 1, 2]
    training_data_ids = [1, 2, 3]
    testing_data_one_hot = [torch.Tensor([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
                            torch.Tensor([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
                            ]
    testing_data_ids = [[1, 1, 3], [1, 2, 0]]
    classifier = classifier_class()
    classifier.train(training_data_ids, ground_truth, classifier_input_size=4, n_classes=4)
    accuracy_on_training_data = classifier.evaluate(training_data_ids, ground_truth, classifier_input_size=4)
    assert accuracy_on_training_data == 1
    for i in range(2):
        assert 0 < classifier.evaluate(testing_data_one_hot[i], ground_truth) \
               == classifier.evaluate(testing_data_ids[i], ground_truth, classifier_input_size=4) \
               <= accuracy_on_training_data


@pytest.mark.parametrize('classifier_class', [SvmClassifier, NNClassifier])
def test_phases(classifier_class: type):
    n_phases = 2
    classifier = classifier_class()
    ground_truth_phases = [[0, 1, 2], [2, 3, 0]]
    input_ids_phases = [[1, 2, 3], [3, 2, 1]]
    input_ids_tensors_phases = [torch.LongTensor(i) for i in input_ids_phases]
    input_tensor_phases = [[torch.Tensor(i) for i in [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
                           [torch.Tensor(i) for i in [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]]]
    accuracy_on_ids = classifier.train_and_evaluate_in_phases(input_ids_phases,
                                                              ground_truth_phases,
                                                              classifier_input_size=4,
                                                              n_classes=4)
    accuracy_on_ids_tensors = classifier.train_and_evaluate_in_phases(input_ids_tensors_phases,
                                                                      ground_truth_phases,
                                                                      classifier_input_size=4,
                                                                      n_classes=4)
    accuracy_on_tensors = classifier.train_and_evaluate_in_phases(input_tensor_phases,
                                                                  ground_truth_phases,
                                                                  n_classes=4)
    assert accuracy_on_ids == accuracy_on_ids_tensors == accuracy_on_tensors == [1.] * n_phases
