from typing import List

import torch
from sklearn import metrics

import numpy as np


def compute_mutual_information_for_phases(
        data: List[List[int]],
        labels: List[List[int]],
        num_classes: int,
        data_contains_id: bool = False,
        normalized: bool = True) -> List[float]:
    """Compute mutual information for testing phases
    Args:
        data: list (for phases) of list of ids
        labels: list of lists of labels
        num_classes: number of classes in the dataset
        data_contains_id: data are scalars?
        normalized: compute normalized MI?

    Returns: List of floats, for each phase one value
    """
    assert len(data) == len(labels)

    results = []

    for phase_id in range(len(data)):
        results.append(compute_mutual_information(np.array(data[phase_id]),
                                                  np.array(labels[phase_id]),
                                                  num_classes,
                                                  data_contains_id=data_contains_id,
                                                  normalized=normalized))
    return results


def compute_mutual_information(
        data: np.array,
        labels: np.array,
        n_classes: int,
        data_contains_id: bool=False,
        normalized: bool = True) -> float:
    """Computes (normalized) mutual information between data and labels.

    Args:
        data_contains_id: data is array of scalars (instead of array of one-hot vectors)?
        data: any no_classes-dimensional output or id of the class
        labels: id of correct label for each observation

    Returns:
        mutual information
    """
    if data_contains_id:
        outputs = data
    else:
        outputs = id_to_one_hot(data, n_classes)

    if len(outputs) != len(labels):
        raise ValueError('Mutual information: length of the samples is different!')

    if normalized:
        result = metrics.normalized_mutual_info_score(labels, outputs)
    else:
        result = metrics.mutual_info_score(labels, outputs)
    # print(f'computing MI between [{",".join(map(str,outputs))}] and [{",".join(map(str,labels))}], result is {result}')
    return result


def id_to_one_hot(data: torch.Tensor, vector_len: int):
    """Convert ID to one-hot representation.

    Args:
        data: ID of the class
        vector_len: max no of classes

    Returns:
        one-hot representation of the data vector
    """

    data_a = data.type(torch.int64).view(-1, 1)
    n_samples = data_a.shape[0]
    output = torch.zeros(n_samples, vector_len)
    output.scatter_(1, data_a, 1)
    return output


def compute_mutual_information_matrix_rfs(cluster_ids: torch.LongTensor) -> torch.Tensor:
    """Computes normalized mutual information for the receptive fields of convolutional experts.

    Args:
        cluster_ids: tensor of cluster indexes with dimensions (n_phases, n_steps, n_flocks, n_rfs_y, n_rfs_x),
        where n_phases is the number of phases, n_steps the number of steps per phase, n_flocks the number of
        convolutional expert groups, and n_rfs_y * n_rfs_x the number of receptive fields.

    Returns:
        Tensor containing, for each phase, pair of convolutional expert groups, and receptive field, the normalized
        mutual information between the cluster ids. It has dimensions (n_phases, n_flocks, n_flocks, n_rfs_y, n_rfs_x).
    """
    n_phases, n_steps, n_flocks, n_rfs_y, n_rfs_x = cluster_ids.shape
    mutual_information_by_rf = torch.zeros(n_phases, n_flocks, n_flocks, n_rfs_y, n_rfs_x)
    for phase in range(n_phases):
        for flock_i in range(n_flocks):
            for flock_j in range(flock_i + 1, n_flocks):
                for rf_y in range(n_rfs_y):
                    for rf_x in range(n_rfs_x):
                        ids_i = cluster_ids[phase, :, flock_i, rf_y, rf_x].cpu().numpy()
                        ids_j = cluster_ids[phase, :, flock_j, rf_y, rf_x].cpu().numpy()
                        mutual_information_by_rf[phase, flock_i, flock_j, rf_y, rf_x] \
                            = metrics.normalized_mutual_info_score(ids_i, ids_j)
    return mutual_information_by_rf


def reduce_to_mean(mutual_information_by_rf: torch.Tensor) -> torch.Tensor:
    """Takes a tensor returned by compute_mutual_information_matrix_rfs and computes the mean over the receptive fields.

    Args:
        mutual_information_by_rf: The tensor returned by compute_mutual_information_matrix_rfs.

    Returns:
        A tensor with dimensions (n_phases, n_flocks, n_flocks). For each phase, it is an upper triangular matrix
        containing the mean normalized mutual information.
    """
    return flatten_rfs(mutual_information_by_rf).mean(dim=-1)


def reduce_to_center_rf(mutual_information_by_rf: torch.Tensor) -> torch.Tensor:
    """Takes a tensor returned by compute_mutual_information_matrix_rfs and gives the MI for the central RF.

    Returns the normalized mutual information for the receptive field at the center of the visual field. If there is
    an even number of receptive fields in either dimension, we take the mean over the central receptive fields.

    Args:
        mutual_information_by_rf: The tensor returned by compute_mutual_information_matrix_rfs.

    Returns:
        A tensor with dimensions (n_phases, n_flocks, n_flocks). For each phase, it is an upper triangular matrix
        containing the normalized mutual information for the central receptive field(s).
    """
    n_phases, n_flocks, n_flocks, n_rfs_y, n_rfs_x = mutual_information_by_rf.shape
    rf_y_start, rf_y_end = center_indexes(n_rfs_y)
    rf_x_start, rf_x_end = center_indexes(n_rfs_x)
    center_slice = mutual_information_by_rf[:, :, :, rf_y_start:rf_y_end, rf_x_start:rf_x_end]
    return flatten_rfs(center_slice).mean(dim=-1)


def center_indexes(index_end: int) -> (int, int):
    """Calculates the indexes defining the center of a range from 0 to index_end.

    Args:
        index_end: End of range.

    Returns:
        Start and end of center sub-range, of length one or two.
    """
    slice_start = index_end // 2 + index_end % 2
    slice_length = 2 if index_end % 2 == 0 else 1
    slice_end = slice_start + slice_length
    return slice_start, slice_end


def flatten_rfs(mutual_information_by_rf: torch.Tensor) -> torch.Tensor:
    """Reshape the tensor returned by compute_mutual_information_matrix_rfs so that RFs are in a single dimension.

    Args:
        mutual_information_by_rf: The tensor returned by compute_mutual_information_matrix_rfs.

    Returns:
        The reshaped tensor.
    """
    n_phases, n_flocks, n_flocks, n_rfs_y, n_rfs_x = mutual_information_by_rf.shape
    return mutual_information_by_rf.view(n_phases, n_flocks, n_flocks, n_rfs_y * n_rfs_x)

