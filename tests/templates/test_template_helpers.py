import pytest
import torch

from torchsim.utils.list_utils import same_lists
from torchsim.utils.template_utils.template_helpers import partition_to_list_of_ids, _partition_tensor_to_ids, \
    compute_derivations


def test_partition_to_list_of_ids():
    """
    Tests the ability to parse
        -list (phase) of list (measurements) of tensors into
        -list (expert_id) of lists (phase) of list (measurements)

    where each value in this List[List[List[int]]] is ID of the active cluster center
    """

    flock_size = 3  # and num_cc = 4

    # measurement 0 in phase 0 (output of the flock at the time step 0)
    meas_0 = [1, 0, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1]

    # measurement 1 in phase 0 (flock output at time step 1)
    meas_1 = [0, 0, 1, 0,
              0, 1, 0, 0,
              1, 0, 0, 0]

    # measurement 0 in phase 1 (flock output at time step 0 at phase 1)
    meas_p_0 = [0, 0, 0, 1,
                1, 0, 0, 0,
                1, 0, 0, 0]

    # measurement 1 in phase 1
    meas_p_1 = [0, 1, 0, 0,
                1, 0, 0, 0,
                0, 1, 0, 0]

    phase0 = [
        torch.Tensor(meas_0).view(flock_size, -1),
        torch.Tensor(meas_1).view(flock_size, -1)]

    phase1 = [
        torch.Tensor(meas_p_0).view(flock_size, -1),
        torch.Tensor(meas_p_1).view(flock_size, -1)]

    # measurements in the format collected by the measurement_manager.parse_to_...
    measurements = [phase0, phase1]

    # test the helper method
    partitioned = _partition_tensor_to_ids(phase0[0], flock_size)
    assert partitioned == [0, 2, 3]

    # test the final method, outer dimension corresponds to experts
    result = partition_to_list_of_ids(measurements, 3)

    # expected measurement for each expert
    e0 = [[0, 2], [3, 1]]  # [[phase_0_measurement_0_id, phase_0_measurement_1_id],[phase_1..., ..]]
    e1 = [[2, 1], [0, 0]]
    e2 = [[3, 0], [0, 1]]

    # dimension for expert is the outer one
    assert result[0] == e0
    assert result[1] == e1
    assert result[2] == e2


@pytest.mark.parametrize('input, first_value, expected_output', [
    ([0, 1, 1, -1, 2], 0.5, [-0.5, 1, 0, -2, 3]),
    ([1, 1.2, 1.2, 0, 0], None, [1, 0.2, 0, -1.2, 0]),
    ([156], 157, [-1])
    ])
def test_compute_derivations(input, first_value, expected_output):

    if first_value is not None:
        result = compute_derivations(input, first_value=first_value)
    else:
        result = compute_derivations(input)

    assert same_lists(expected_output, result, eps=1e-4)

