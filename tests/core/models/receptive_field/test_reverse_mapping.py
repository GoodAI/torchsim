import pytest

import torch

from torchsim.core import get_float, FLOAT_NAN
from torchsim.core.models.receptive_field.grid import Grids, Stride
from torchsim.utils.param_utils import Size2D
from torchsim.core.models.receptive_field.reverse_mapping import ReverseMapping
from torchsim.core.utils.tensor_utils import same


class TestReverseMapping:
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_reverse_map_concat(self, device):
        # device = "cpu"
        float_dtype = get_float(device)

        def context(part: int, parent_id: int):
            context_size = 5
            start = 2 * parent_id + part
            return torch.arange(start * context_size, (start + 1) * context_size, dtype=float_dtype,
                                device=device)

        def nans(count: int):
            return torch.full((count,), FLOAT_NAN, device=device)

        data_size = 2 * 5
        grids = Grids(Size2D(3, 4), parent_rf_dims=Size2D(2, 3), parent_rf_stride=Stride(1, 1), device=device)
        assert 4 == len(list(grids.gen_parent_receptive_fields()))
        mapping = ReverseMapping(grids, device, data_size)
        data = torch.arange(4 * 2 * 5, device=device).view(2, 2, 2, 5).float()
        result = mapping.reverse_map_concat(data)
        assert (3, 4, 2, 3, 2, 5) == result.shape
        # r = result.view(12, 6, 10)
        # r = result.view(12, 6, 2, 5)
        # r = r.transpose(1, 2).contiguous()
        r = result.view(12, 6, 2, 5)

        for part in [0, 1]:
            child = 0
            assert same(nans(25), r[child, 0:5, part, :].contiguous().view(-1)), f'Part {part}'
            assert context(part, 0).equal(r[child, 5, part, :]), f'Part {part}'

            child = 1
            assert same(nans(20), r[child, 0:4, part,  :].contiguous().view(-1)), f'Part {part}'
            assert context(part, 0).equal(r[child, 4, part, :]), f'Part {part}'
            assert context(part, 1).equal(r[child, 5, part, :]), f'Part {part}'

            child = 2
            assert same(nans(15), r[child, 0:3, part, :].contiguous().view(-1)), f'Part {part}'
            assert context(part, 0).equal(r[child, 3, part, :]), f'Part {part}'
            assert context(part, 1).equal(r[child, 4, part, :]), f'Part {part}'
            assert same(nans(5), r[child, 5, part, :].contiguous().view(-1)), f'Part {part}'

            child = 3
            assert same(nans(15), r[child,  0:3, part, :].contiguous().view(-1)), f'Part {part}'
            assert context(part, 1).equal(r[child, 3, part, :]), f'Part {part}'
            assert same(nans(10), r[child, 4:6, part, :].contiguous().view(-1)), f'Part {part}'

            child = 4
            assert same(nans(10), r[child, 0:2, part, :].contiguous().view(-1)), f'Part {part}'
            assert context(part, 0).equal(r[child, 2, part, :]), f'Part {part}'
            assert same(nans(10), r[child,  3:5, part, :].contiguous().view(-1)), f'Part {part}'
            assert context(part, 2).equal(r[child, 5, part, :]), f'Part {part}'

            child = 5
            assert same(nans(5), r[child, 0, part, :].contiguous().view(-1)), f'Part {part}'
            assert context(part, 0).equal(r[child, 1, part, :]), f'Part {part}'
            assert context(part, 1).equal(r[child, 2, part, :]), f'Part {part}'

            assert same(nans(5), r[child, 3, part, :].contiguous().view(-1)), f'Part {part}'
            assert context(part, 2).equal(r[child, 4, part, :]), f'Part {part}'
            assert context(part, 3).equal(r[child, 5, part, :]), f'Part {part}'
