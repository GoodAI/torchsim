import pytest
import torch
import time


@pytest.mark.skip(reason="benchmarking purposes only")
def test_indexing():
    data = torch.rand((10000, 500, 4)).to('cuda')
    indices = torch.rand((10000, 500)).to('cuda')

    indices[indices < 0.5] = 0

    torch.cuda.synchronize()
    start = time.clock()
    data[indices == 0] = -1
    duration = time.clock() - start
    torch.cuda.synchronize()

    print(duration)


@pytest.mark.skip(reason="benchmarking purposes only")
def test_indexing_2():

    data = torch.rand((10000, 500, 4)).to('cuda')
    indices = torch.rand((10000, 500)).to('cuda')

    indices[indices < 0.5] = 0

    torch.cuda.synchronize()
    start = time.clock()
    data *= indices.unsqueeze(dim=2)
    torch.cuda.synchronize()
    duration = time.clock() - start

    print(duration)
