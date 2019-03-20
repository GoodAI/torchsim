import random

import pytest
from pytest import raises
from torchsim.core.memory.tensor_creator import MeasuringCreator, AllocatingCreator, TensorSurrogate, TensorCreator


def _tensor_from_data(creator: TensorCreator):
    return creator.tensor([1, 2, 3], dtype=creator.float64, device='cpu')


def _zeros(creator: TensorCreator):
    return creator.zeros((6, 5, 2), device='cpu')


def _zeros_single_dim(creator: TensorCreator):
    return creator.zeros(6, device='cpu')


def _full(creator: TensorCreator):
    return creator.full((2, 3, 1, 1), fill_value=-3)


def _ones(creator: TensorCreator):
    return creator.ones((3, 2, 4), dtype=creator.int64)


def _ones_single_dim(creator: TensorCreator):
    return creator.ones(6, device='cpu')


def _empty(creator: TensorCreator):
    return creator.ones((3, 2, 4), dtype=creator.int64)


def _cat_dim_0(creator: TensorCreator):
    return creator.cat([creator.zeros((1, 3)), creator.ones((3, 3))], dim=0)


def _cat_dim_1(creator: TensorCreator):
    return creator.cat([creator.zeros((3, 1)), creator.ones((3, 3))], dim=1)


def _eye_default(creator: TensorCreator):
    return creator.eye(4)


def _eye_non_default(creator: TensorCreator):
    return creator.eye(4, 3)


def _arange_divisible(creator: TensorCreator):
    return creator.arange(0, end=3, step=2)


def _arange_non_divisible(creator: TensorCreator):
    return creator.arange(0, end=4, step=2)


@pytest.mark.parametrize('create_func', [
    _tensor_from_data,
    _zeros,
    _zeros_single_dim,
    _full,
    _ones,
    _ones_single_dim,
    _empty,
    _cat_dim_0,
    _cat_dim_1,
    _eye_default,
    _eye_non_default,
    _arange_divisible,
    _arange_non_divisible
])
def test_creator_methods(create_func):
    measuring_creator = MeasuringCreator()
    allocating_creator = AllocatingCreator('cpu')

    expected = create_func(allocating_creator)
    surrogate = create_func(measuring_creator)

    assert expected.shape == surrogate.shape


@pytest.mark.parametrize('creator', [MeasuringCreator(), AllocatingCreator('cpu')])
class TestTensorSurrogate:

    @pytest.mark.parametrize('shape, view, expected_result', [
        ((2, 3), (1, 6), (1, 6)),
        ((2, 3), (1, -1), (1, 6)),
        ((2, 3), (3, -1), (3, 2)),
    ])
    def test_view(self, shape, view, creator, expected_result):
        tensor = creator.zeros(shape)
        assert expected_result == tensor.view(view).shape
        assert expected_result == tensor.view(view).shape, 'check that original tensor is not modified'

    @pytest.mark.parametrize('shape, view, expected_result', [
        ((2, 3), (1, 6), (1, 6)),
        ((2, 3), (1, -1), (1, 6)),
        ((2, 3), (3, -1), (3, 2)),
    ])
    def test_reshape(self, shape, view, creator, expected_result):
        tensor = creator.zeros(shape)
        assert expected_result == tensor.reshape(view).shape
        assert expected_result == tensor.reshape(view).shape, 'check that original tensor is not modified'

    @pytest.mark.parametrize('shape, expand, expected_result', [
        ((1, 3, 1), (2, 3, 5), (2, 3, 5)),
        ((1, 3, 1), (3, -1, 2), (3, 3, 2)),
    ])
    def test_expand(self, shape, expand, creator, expected_result):
        tensor = creator.zeros(shape)
        assert expected_result == tensor.expand(expand).shape
        assert expected_result == tensor.expand(expand).shape, 'check that original tensor is not modified'

    @pytest.mark.parametrize('shape, expected_result', [
        ((2, 2, 2, 2, 2, 2), 6),
        ((1, 2, 3), 3),
        ((1,), 1),
        ((), 0),
    ])
    def test_dim(self, shape, creator, expected_result):
        tensor = creator.zeros(shape)
        assert expected_result == tensor.dim()

    @pytest.mark.parametrize('tensor_shape, index_shape, dim, expected_shape', [
        ((2, 3), (2, 5), 1, (2, 5)),
        ((2, 3), (1, 3), 0, (1, 3)),
        ((2, 3, 5, 8), (1, 3, 5, 8), 0, (1, 3, 5, 8)),
        ((2, 3, 5, 8), (2, 1, 5, 8), 1, (2, 1, 5, 8)),
        ((2, 3, 5, 8), (2, 3, 1, 8), 2, (2, 3, 1, 8)),
        ((2, 3, 5, 8), (2, 3, 5, 1), 3, (2, 3, 5, 1)),
    ])
    def test_gather(self, creator, tensor_shape, index_shape, dim, expected_shape):
        tensor = creator.zeros(tensor_shape)
        index = creator.zeros(index_shape).long()
        assert expected_shape == tensor.gather(dim, index).shape

    @pytest.mark.parametrize('tensor_shape, index_shape, dim, expected_shape', [
        ((2, 3), (2, 5), 0, (2, 5)),
        ((2, 3), (2, 5), 2, (2, 5)),
        ((2, 1, 5, 8), (2, 3, 5, 8), 0, (2, 3, 5, 8)),
        ((2, 3, 1, 8), (2, 3, 5, 8), 0, (2, 3, 5, 8)),
        ((2, 3, 5, 1), (2, 3, 5, 8), 0, (2, 3, 5, 8)),
    ])
    def test_gather_error_input(self, creator, tensor_shape, index_shape, dim, expected_shape):
        with raises(RuntimeError):
            tensor = creator.zeros(tensor_shape)
            index = creator.zeros(index_shape).long()
            assert expected_shape == tensor.gather(dim, index).shape

    @pytest.mark.parametrize('input_shape, dim', [
        ([2, 3, 1, 4], 0),
        ([2, 3, 1, 4], 2),
        ([2, 3, 1, 4], 3),
        ([9, 15, 0, 6], 4),
        ([2, 3, 1, 4], -1),
        ([2, 3, 1, 4], -4)
    ])
    def test_unsqueeze(self, creator, input_shape, dim):
        dtype = random.randint(0, 1000)
        device = random.randint(0, 1000)

        input_tensor_surrogate = TensorSurrogate(input_shape, dtype=dtype, device=device)
        input_tensor_real = creator.zeros(input_shape)

        result = input_tensor_surrogate.unsqueeze(dim)
        expected_result = input_tensor_real.unsqueeze(dim)

        assert expected_result.shape == result.shape
        # check that it preserves dtype and device
        assert dtype == result.dtype
        assert device == result.device
