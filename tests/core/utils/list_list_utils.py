import pytest
import torch

from torchsim.core.utils.tensor_utils import same
from torchsim.utils.list_utils import flatten_list, same_lists


def test_flatten_list():
    list_of_lists = [[], [1, 2, 3], [0, 0], [], [], [5, 6, 7, 8], []]

    result = flatten_list(list_of_lists)
    expected_result = [1, 2, 3, 0, 0, 5, 6, 7, 8]

    assert result == expected_result


@pytest.mark.parametrize('list1, list2, eps, expected_result', [
    ([1, 1.3, -15, 0], [1, 1.3, -15, 0], None, True),
    ([1, 1.3, -15, float('nan')], [1, 1.3, -15, 0], None, False),
    ([float('nan'), 0], [float('nan'), 0], None, True),
    ([0, float('nan')], [float('nan'), 0], None, False),
    ([], [], None, True),
    ([0.9999, 1.39, -15.356], [1, 1.3, -15.356], 0.01, False),
    ([0.9999, 1.39, -15.356], [1, 1.3, -15.356], 0.1, True)
])
@pytest.mark.parametrize('comparison_method', ([same_lists, same]))
def test_same_list(comparison_method, list1, list2, eps, expected_result):
    """This test tst together two util methods for comparing: same() for tensors and same_lists for lists."""

    if comparison_method == same_lists:
        input1 = list1
        input2 = list2
    else:
        input1 = torch.tensor(list1)
        input2 = torch.tensor(list2)

    result = comparison_method(input1, input2, eps=eps)

    assert expected_result == result
