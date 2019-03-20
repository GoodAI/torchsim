import torch

from torchsim.core.eval.metrics.mutual_information_metric import compute_mutual_information_matrix_rfs, reduce_to_mean, \
    reduce_to_center_rf


def test_cluster_mutual_information():
    # Input:
    # Tensor with indexes [phase, step, flock, rf_y, rf_x]. A flock corresponds to a convolutional group of experts.
    # We assume the number of steps is the same for all the phases.
    #
    # Intermediate structure:
    # Normalized mutual information by RF, indexed [phase, flock_i, flock_j, rf_y, rf_x] where flock_i < flock_j
    #
    # Output:
    # Upper triangular matrices of normalized MI for each phase and pair of flocks, indexed [phase, flock_i, flock_j],
    # flock_i < flock_j.
    # Each MI value computed from the MIs for each RF: mean, min, max, center RF

    n_phases = 1
    n_steps = 5
    n_flocks = 3
    n_rfs_y, n_rfs_x = 2, 3

    cluster_ids = torch.zeros(n_phases, n_steps, n_flocks, n_rfs_y, n_rfs_x, dtype=torch.long)

    # Make flocks 0 and 1 agree perfectly, put something different in flock 2
    for step in range(n_steps):
        id_2 = step % 2
        id_3 = step % 3
        for flock in [0, 1]:
            cluster_ids[0, step, flock, 0, 0] = id_3
        cluster_ids[0, step, 2, 0, 0] = id_2

    # Normalized mutual information by RF, indexed [phase, flock, flock, rf]
    mutual_information_by_rf = compute_mutual_information_matrix_rfs(cluster_ids)

    mean_matrix = reduce_to_mean(mutual_information_by_rf)
    # max_matrix = mutual_information_by_rf.max(dim=-1)[0]
    # min_matrix = mutual_information_by_rf.min(dim=-1)[0]
    center_matrix = reduce_to_center_rf(mutual_information_by_rf)

    # Dimensions
    assert cluster_ids.shape == (n_phases, n_steps, n_flocks, n_rfs_y, n_rfs_x)
    assert mutual_information_by_rf.shape == (n_phases, n_flocks, n_flocks, n_rfs_y, n_rfs_x)
    assert mean_matrix.shape == center_matrix.shape == (n_phases, n_flocks, n_flocks)

    # Diagonal entries, should always be the same
    diagonal_value = 0
    for phase in range(n_phases):
        for flock in range(n_flocks):
            for matrix in [mean_matrix, center_matrix]:
                assert matrix[phase, flock, flock] == diagonal_value

    # Entries below the diagonal, always the same
    value_below_diagonal = 0
    for phase in range(n_phases):
        for flock_i in range(n_flocks):
            for flock_j in range(flock_i):
                for matrix in [mean_matrix, center_matrix]:
                    assert matrix[phase, flock_i, flock_j] == value_below_diagonal

    # RFs other than (0, 0) have normalized MI == 1
    for phase in range(n_phases):
        for flock_i in range(n_flocks):
            for flock_j in range(flock_i + 1, n_flocks):
                for rf_y in range(n_rfs_y):
                    for rf_x in range(1, n_rfs_x):
                        assert mutual_information_by_rf[phase, flock_i, flock_j, rf_y, rf_x] == 1

    # Perfect agreement, normalized MI == 1
    assert mean_matrix[0, 0, 1] == 1

    # Less than perfect agreement, normalized MI < 1
    assert 0 < mean_matrix[0, 0, 2] < 1
