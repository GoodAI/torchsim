import logging
from typing import List, Union

import numpy as np
import torch

from torchsim.core.eval.metrics.simple_classifier_nn import compute_nn_classifier_accuracy

logger = logging.getLogger(__name__)


def compute_mse_from(a: List[torch.Tensor], b: List[torch.Tensor]):
    mse = 0
    for first, second in zip(a, b):
        if first is None or second is None:
            mse += 0
            logger.error(f'got None on one of the inputs!')
        else:
            mse += torch.sum((first - second) ** 2) / first.numel()

    return mse / len(a)


def compute_classification_accuracy(labels: List[int], predictions: List[int], num_classes: int = -1) -> float:
    """ Computes the classification accuracy from IDs; accuracy = correct/all
    Args:
        labels:
        predictions:
        num_classes: not used (just implements the same "interface" as the cmpute_se_classification_accuracy)
    Returns: accuracy from [0,1]
    """
    assert len(labels) == len(predictions)

    correct = 0
    for a, b in zip(labels, predictions):
        if a == b:
            correct += 1

    return correct / len(labels)


def compute_se_classification_accuracy(labels: List[int], predictions: List[int], num_classes: int) -> float:
    """ Classification accuracy according to the SE tasks (with aggregation)
    
    Computes the classifier accuracy as the SE task does:
    For each object (sequence of identical labels) compute histogram of output IDs.
    The answer of the architecture is the most frequent answer in this window.
    Accuracy is num of correct answers/num of incorrect ones in the window.

    Note that this is just an approximation! (object ids not available yet); TODO ?
    Args:
        num_classes: number of classes in the dataset
        labels: lit of labels (scalars)
        predictions: predicted classes (scalars)
    Returns: SE task 0-compatible classifier accuracy
    """
    assert len(labels) == len(predictions)

    pos = 0
    encountered_objects = 0
    correct_answers = 0

    while pos < len(labels):
        correct_answer = labels[pos]
        pos, answer_id = _collect_answers_for(labels,
                                              predictions,
                                              correct_answer,
                                              pos,
                                              num_classes)
        encountered_objects += 1
        if answer_id == correct_answer:
            correct_answers += 1

    return correct_answers / encountered_objects


def _collect_answers_for(labels: List[int],
                         predictions: List[int],
                         label: int,
                         start_pos: int,
                         num_classes: int) -> [int, int]:
    answers_histogram = np.zeros(num_classes)
    pos = start_pos

    for a, b in zip(labels[start_pos:], predictions[start_pos:]):
        # if we encountered different object (different label ~ the approximation), return position and answer
        if a != label:
            return pos, np.argmax(answers_histogram)
        pos += 1
        answers_histogram[b] += 1

    return pos, np.argmax(answers_histogram)


def argmax_list_list_tensors(tensors: List[List[torch.Tensor]]) -> List[List[int]]:
    """ Convert list of tensors to list of their argmaxes

    Args:
        tensors: List[List[tensor]] outputs of the classifier for each phase
    Returns: list of lists of output ids (argmax)
    """
    result = []
    for phase in tensors:
        ids_in_phase = []
        for tensor in phase:
            ids_in_phase.append(argmax_tensor(tensor))
        result.append(ids_in_phase)

    return result


def argmax_tensor(data: torch.Tensor) -> int:
    assert len(data.shape) == 1 or data.shape[0] == 1

    data = data.view(-1)
    _, arg_max = data.max(0)
    return int(arg_max.to('cpu').item())


def compute_nn_classifier_accuracy_in_phases(outputs: List[List[int]],
                                             labels: List[List[int]],
                                             classifier_input_size: int,
                                             num_classes: int,
                                             device: str = 'cpu') -> List[float]:
    """ Compute weak classifier accuracy

    TODO this is now obsolete, see the AbstractClassifier

    Args:
        classifier_input_size: size of the input layer of the classifier = size of the output of the model
        outputs: outputs of the model
        labels: ground-truth labels
        num_classes: number of classes in the dataet
        device: cuda/cpu
    Returns: list of accuracies, one for one phase
    """

    results = []
    for phase_outputs, phase_labels in zip(outputs, labels):
        acc = do_compute_nn_classifier_accuracy(phase_outputs,
                                                phase_labels,
                                                classifier_input_size=classifier_input_size,
                                                num_classes=num_classes,
                                                device=device)
        results.append(acc)

    return results


def do_compute_nn_classifier_accuracy(outputs: Union[torch.Tensor, List[int]],
                                      labels: Union[torch.Tensor, List[int]],
                                      classifier_input_size: int,
                                      num_classes: int,
                                      device: str = 'cpu') -> float:
    """ Computes classifier accuracy in the template.

    compute the classifier accuracy trained on the outputs (scalar format from the range <0,output_dim)
    to predict the labels (scalar format from the range <0,n_classes)
    Args:
        device: cuda/cpu
        num_classes: number of classes in the dataset
        outputs: either torch.LongTensor of size [n_samples] or List of ints (scalars)
        labels: either torch.LongTensor of size [n_samples] or List of ints
        classifier_input_size: range of ids in the outputs tensor
    Returns: low-VC dimension classifier accuracy trained and tested on this data
    """

    if type(outputs) is list:
        outputs = torch.tensor(outputs, device=device, dtype=torch.long)
    if type(labels) is list:
        labels = torch.tensor(labels, device=device, dtype=torch.long)

    if is_containing_nans(outputs):
        logger.error(f'outputs is containing NaNs')
    if is_containing_nans(labels):
        logger.error(f'labels is containing NaNs')

    acc = compute_nn_classifier_accuracy(outputs,
                                         labels,
                                         n_classes=num_classes,
                                         custom_input_length=classifier_input_size,
                                         log_loss=False,
                                         max_epochs=50)
    return acc


def is_containing_nans(input_tensor: torch.Tensor):
    if torch.isnan(input_tensor).any():
        return True
    return False


def compute_label_reconstruction_accuracies(ground_truth_ids: List[List[int]],
                                            baseline_output_tensors: List[List[torch.Tensor]],
                                            model_output_tensors: List[List[torch.Tensor]],
                                            accuracy_method,
                                            num_classes: int) -> (List[List[float]], List[List[float]]):
    """ Compute accuracy of classification of the model and the (random) baseline in the test phases.

    Computes over different testing phases (first list always corresponds to testing phases)

    Args:
        ground_truth_ids: list (each phase) of lists (measurements) of scalar label IDs
        baseline_output_tensors: torch tensors corresponding to outputs during testing
        model_output_tensors: outputs of the model
        accuracy_method: which accuracy to compute? normal or SE-aggregated?
        num_classes: have to provide num. classes (i.e. 20 in this case)

    Returns: baseline accuracy and model accuracy

    """

    base_acc = _compute_classification_accuracy(
        label_ids=ground_truth_ids,
        output_tensors=baseline_output_tensors,
        accuracy_method=accuracy_method,
        num_classes=num_classes
    )

    model_acc = _compute_classification_accuracy(
        label_ids=ground_truth_ids,
        output_tensors=model_output_tensors,
        accuracy_method=accuracy_method,
        num_classes=num_classes
    )
    return base_acc, model_acc


def _compute_classification_accuracy(label_ids: List[List[int]],
                                     output_tensors: List[List[torch.Tensor]],
                                     accuracy_method,
                                     num_classes: int) -> List[List[float]]:
    """ Compute classification accuracy for collected outputs
    Args:
        label_ids: List (for testing phase) of Lists of ints (id ~ each measurement in the phase)
        output_tensors: List (for testing phase) of Lists of tensors (tensor ~ output measured)

    Returns: List (for testing phase) of floats (accuracy)
    """
    output_ids = argmax_list_list_tensors(output_tensors)
    assert len(output_ids) == len(label_ids)
    phase_accuracies = []
    for phase_label_ids, phase_output_ids in zip(label_ids, output_ids):
        phase_accuracies.append(accuracy_method(phase_label_ids, phase_output_ids, num_classes=num_classes))
    return phase_accuracies


def compute_mse_values(labels: List[List[torch.Tensor]], outputs: List[List[torch.Tensor]]) -> List[float]:
    """ Compute the MSE between labels and given outputs.

    Computed for lists of testing phases.

    Args:
        labels: List (testing phases) of lists (measurements) of torch tensors
        outputs: the same, outputs of the model/baseline

    Returns: list of MSE values, one for each testing phase

    """
    mse_values = []
    for labels_per_phase, outputs_per_phase in zip(labels, outputs):
        mse_values.append(compute_mse_from(labels_per_phase, outputs_per_phase).item())
    return mse_values


def list_int_to_long_tensors(model_outputs: List[List[int]]) -> List[torch.Tensor]:
    """ Convert the List (for testing phase) of Lists of scalars (measurements) to List of LongTensors.
    Args:
        model_outputs: model outputs splitted into the testing phases
    Returns: list of LongTensors, for each testing phase one tensor (containing all scalar outputs from the phase)
    """
    result = []

    # TODO shorten with list comprehension here ideally
    for single_phase_outputs in model_outputs:
        result.append(torch.tensor(single_phase_outputs, dtype=torch.long))

    return result


def _partition_tensor_to_ids(tensor: torch.Tensor, flock_size: int) -> List[int]:
    """ Partition the torch tensor into tensor.numel()//flock_size parts, find argmax of each part
    Args:
        tensor: sp output with flock_size>1
        flock_size: flock size!
    Returns: list of argmax ids (cluster center ids)
    """
    assert tensor.numel() % flock_size == 0

    argmaxes = []

    for expert_id in range(flock_size):
        items = tensor[expert_id].tolist()
        max_id = items.index(max(items))  # python argmax
        argmaxes.append(max_id)

    return argmaxes


def partition_to_list_of_ids(tensors: List[List[torch.Tensor]], flock_size: int) -> List[List[List[int]]]:
    """Partition the output tensor of the flock with flock_size>1 into List of ids for each expert

    For an example see the \tests\templates\test_template_helpers.py\test_partition_to_list_of_ids()

    Args:
        tensors: list (phase) of list (measurements) of tensors (one tensor of shape [flock_size, num_cc]
        flock_size: num experts in the flock

    Returns: list (expert_id) of lists (phase) of list (measurements), outer List corresponds to experts
    """

    # list of experts
    results = []

    # for each phase, for each measurement, split the tensor to list of ids
    for phase_id, phase in enumerate(tensors):
        results.append([])  # append the next phase

        for measurement_id, measurement in enumerate(phase):
            ids = _partition_tensor_to_ids(measurement, flock_size)
            results[-1].append(ids)  # append list of expert output ids

    # invert order of the dimensions from [phase][measurements][experts] to [experts][phase][measurements]
    r = []
    for expert_id, _ in enumerate(results[0][0]):
        r.append([])  # append expert_id
        for phase_id, phase in enumerate(results):
            r[expert_id].append([])  # phase_id
            for measurement in results[phase_id]:
                r[expert_id][phase_id].append(measurement[expert_id])

    return r


def compute_derivations(values: List[float], first_value: float=0.) -> List[float]:
    """ Computes differences between two succeeding floats in list.

    Args:
        values: list of floats
        first_value: total number of derivations is shorter of one element, so this value and the first value
         in the list will be used for computation of one extra derivation on the start. You can omit it using [1:].

    Returns:
        List of derivations. Size is the same as for values.
    """
    return [v1 - v0 for v0, v1 in zip([first_value] + values, values)]
