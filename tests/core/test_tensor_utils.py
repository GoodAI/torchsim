import math

import pytest
import torch

from torchsim.core import get_float, FLOAT_NAN
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.utils.tensor_utils import same, move_probs_towards_50_, move_probs_towards_50, normalize_probs, negate, \
    gather_from_dim, id_to_one_hot, kl_divergence, view_dim_as_dims, safe_id_to_one_hot, clamp_tensor


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
def test_move_probabilities_towards_50_(device):
    float_type = get_float(device)

    input = torch.tensor([[[1, 0, 0],
                           [1, 0, 0]],
                          [[1, 0.5, 0.5],
                           [0.3333, 0.6666, 0.9]]],
                         dtype=float_type, device=device)

    expected_output = torch.tensor([[[0.9999, 0.0001, 0.0001],
                                     [0.9999, 0.0001, 0.0001]],
                                    [[0.9999, 0.5, 0.5],
                                     [0.333333, 0.666567, 0.89992]]],
                                   dtype=float_type, device=device)

    move_probs_towards_50_(input)

    assert same(expected_output, input, eps=1e-2)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
def test_move_probabilities_towards_50(device):
    float_type = get_float(device)

    input = torch.tensor([[[1, 0, 0],
                           [1, 0, 0]],
                          [[1, 0.5, 0.5],
                           [0.3333, 0.6666, 0.9]]],
                         dtype=float_type, device=device)

    expected_output = torch.tensor([[[0.9999, 0.0001, 0.0001],
                                     [0.9999, 0.0001, 0.0001]],
                                    [[0.9999, 0.5, 0.5],
                                     [0.333333, 0.666567, 0.89992]]],
                                   dtype=float_type, device=device)

    output = move_probs_towards_50(input)

    assert same(expected_output, output, eps=1e-2)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
def test_move_probabilities_towards_50_inplace_vs_not(device):
    float_type = get_float(device)

    input = torch.rand((1, 2, 3, 4),
                       dtype=float_type, device=device)

    not_inplace = move_probs_towards_50(input)
    move_probs_towards_50_(input)

    assert same(input, not_inplace, eps=1e-4)


def test_normalize_probabilities():
    probabilities = torch.zeros(20)
    normalized = normalize_probs(probabilities, 0, add_constant=True)
    assert math.isclose(torch.sum(normalized).item(), 1, rel_tol=1e-5)
    probabilities = torch.ones(10)
    normalized = normalize_probs(probabilities, 0, add_constant=True)
    assert math.isclose(torch.sum(normalized).item(), 1, rel_tol=1e-5)
    normalized = normalize_probs(probabilities, 0, add_constant=False)
    assert math.isclose(torch.sum(normalized).item(), 1, rel_tol=1e-5)
    normalized = normalize_probs(normalized, 0, add_constant=True)
    assert math.isclose(torch.sum(normalized).item(), 1, rel_tol=1e-5)
    normalized = normalize_probs(normalized, 0, add_constant=False)
    assert math.isclose(torch.sum(normalized).item(), 1, rel_tol=1e-5)


def test_negate():
    zeros = torch.zeros(10)
    ones = torch.ones(10)
    non_zeros = torch.Tensor([1, -2, .4, 12314, -1e5, 1e-4, 1, 3, 4, -100])
    assert zeros.equal(negate(ones).type_as(zeros))
    assert ones.equal(negate(zeros).type_as(ones))
    assert zeros.equal(negate(non_zeros).type_as(zeros))


def test_same():
    """Another test is together with same_list in test_list_utils."""
    assert same(torch.ones(10), torch.ones(10), 1e-5)
    assert not same(torch.ones(10), torch.zeros(10), 1e-5)


@pytest.mark.parametrize("source, indices, dim, raises_exception", [
    ([[1, 2, 3], [4, 5, 6]], [2, 1], 1, False),  # expected: [[3,2], [6,5]]
    ([[1, 2, 3], [4, 5, 6]], [0, 1, 2, 1], 1, False),
    ([[1, 2, 3], [4, 5, 6]], [1], 1, False),
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [0, 2], 0, False),
    ([[[1, 2, 3], [4, 5, 6]], [[9, 8, 7], [6, 5, 4]]], [[1, 1]], 1, True),
    ([[1, 2, 3], [4, 5, 6]], [[2, 1], [2, 1]], 1, True),
    ([[[[[[[1]]]]]]], [0], 5, False)
])
def test_gather_from_dim(source, indices, dim, raises_exception):
    source = torch.tensor(source)
    indices = torch.tensor(indices, dtype=torch.int64)

    if raises_exception:
        with pytest.raises(ValueError):
            gather_from_dim(source, indices, dim=dim)
    else:
        # we want to do smart indexing over the dim dimension. For example for dim = 3: source[:, :, :, indices]
        # to achieve that, we need to use a slice object instead of each colon:
        slices = [slice(None)] * dim
        slices += [indices]
        # so slices now look like this: [slice(), slice(), slice(), indices]

        expected = source[slices]
        res = gather_from_dim(source, indices, dim=dim)
        assert same(expected, res)


@pytest.mark.parametrize('indexes, number_of_elements, dtype, expected_output', [
    (
            [[3, 2, 4], [0, 0, 1]], 5, torch.int64,
            [[[0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1]],
             [[1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0]]]
    ), (
            [[1, 0]], 3, torch.float32,
            [[[0, 1, 0],
              [1, 0, 0]]]
    ),
    ([0], 1, torch.int8, [[1]]),
    ([1], 2, None, [[0, 1]]),
    ([0], 4, None, [[1, 0, 0, 0]])
])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
def test_id_to_one_hot(indexes, number_of_elements, dtype, expected_output, device):
    float_type = get_float(device)
    indexes_tensor = torch.tensor(indexes, dtype=torch.int64, device=device)

    if dtype is None:  # test default dtype
        result = id_to_one_hot(indexes_tensor, number_of_elements)
        ground_truth = torch.tensor(expected_output, dtype=float_type, device=device)
    else:
        result = id_to_one_hot(indexes_tensor, number_of_elements, dtype=dtype)
        ground_truth = torch.tensor(expected_output, dtype=dtype, device=device)

    assert same(ground_truth, result)


@pytest.mark.parametrize('indexes, number_of_elements, dtype, expected_output', [
    (
            [[3, -1, 4], [0, 0, 1]], 5, torch.int64,
            [[[0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1]],
             [[1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0]]]
    ), (
            [[1, -1]], 3, torch.float32,
            [[[0, 1, 0],
              [0, 0, 0]]]
    ),
    ([[-1, 0, -1], [-1, -1, -1]], 1, torch.int8, [[[0], [1], [0]], [[0], [0], [0]]]),
    ([1], 2, None, [[0, 1]]),
    ([0], 4, None, [[1, 0, 0, 0]]),
    ([4], 4, None, [[0, 0, 0, 0]]),
    ([-2], 4, None, [[0, 0, 0, 0]])
])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
def test_safe_id_to_one_hot(indexes, number_of_elements, dtype, expected_output, device):
    float_type = get_float(device)
    indexes_tensor = torch.tensor(indexes, dtype=torch.int64, device=device)

    if dtype is None:  # test default dtype
        result = safe_id_to_one_hot(indexes_tensor, number_of_elements)
        ground_truth = torch.tensor(expected_output, dtype=float_type, device=device)
    else:
        result = safe_id_to_one_hot(indexes_tensor, number_of_elements, dtype=dtype)
        ground_truth = torch.tensor(expected_output, dtype=dtype, device=device)

    assert same(ground_truth, result)


@pytest.mark.parametrize('clamped_data, min, max, expected_result', [
    (
            [[0.36, -0.48, 0.16], [-0.333, -0.333, 0.333]],
            [[-0.1], [-0.2]],
            [[0.36], [-0.1]],
            [[0.36, -0.1, 0.16], [-0.2, -0.2, -0.1]]
    ), (
            [[[1, 2, 3, 4]]],
            None,
            None,
            [[[1, 2, 3, 4]]],
    ), (
            [[[1, 2, 3, 4]]],
            [[[2.5]]],
            None,
            [[[2.5, 2.5, 3, 4]]],
    ), (
            [[[1, 2, 3, 4]]],
            None,
            [[[2.1]]],
            [[[1, 2, 2.1, 2.1]]],
    )
])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_clamp_tensor(clamped_data, min, max, expected_result, device):
    dtype = get_float(device)
    creator = AllocatingCreator(device=device)
    clamped_tensor = creator.tensor(clamped_data, dtype=dtype, device=device)
    min_tensor = None if min is None else creator.tensor(min, dtype=dtype, device=device)
    max_tensor = None if max is None else creator.tensor(max, dtype=dtype, device=device)

    expected_tensor = creator.tensor(expected_result, dtype=dtype, device=device)

    result = clamp_tensor(clamped_tensor, min_tensor, max_tensor)

    assert same(expected_tensor, result)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
@pytest.mark.parametrize('p, q, expected_result', [
    (
            [
                [0.36, 0.48, 0.16],
                [0.333, 0.333, 0.333],
            ],
            [
                [0.333, 0.333, 0.333],
                [0.36, 0.48, 0.16],
            ],
            [0.0863, 0.0964]
    )
])
def test_kl_divergence(device, p, q, expected_result):
    float_type = get_float(device)
    t_p = torch.tensor(p, dtype=float_type, device=device)
    t_q = torch.tensor(q, dtype=float_type, device=device)
    t_expected_result = torch.tensor(expected_result, dtype=float_type, device=device)
    t_result = torch.zeros((t_p.shape[0],), device=device)
    kl_divergence(t_p, t_q, t_result, dim=1)
    assert same(t_expected_result, t_result, eps=1e-4)


@pytest.mark.parametrize('tensor_shape, shape, dim, expected_shape', [
    ((1, 5, 4), (2, 2), 2, (1, 5, 2, 2)),
    ((1, 4, 5), (2, 2), 1, (1, 2, 2, 5)),
    ((4, 1, 5), (2, 2), 0, (2, 2, 1, 5)),
    ((1, 5, 4), (2, 2), -1, (1, 5, 2, 2)),
    ((1, 4, 5), (2, 2), -2, (1, 2, 2, 5)),
    ((4, 1, 5), (2, 2), -3, (2, 2, 1, 5))
])
def test_view_dim_as_dims(tensor_shape, shape, dim, expected_shape):
    result = view_dim_as_dims(torch.zeros(tensor_shape), shape, dim)
    assert expected_shape == result.shape
