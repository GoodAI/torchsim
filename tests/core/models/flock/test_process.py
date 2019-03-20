import pytest
import torch

from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.flock import Process
from torchsim.core.models.flock.buffer import Buffer
from torchsim.core.utils.tensor_utils import same


class DummyBuffer(Buffer):
    def __init__(self, creator, flock_size, buff_size, dims):
        super().__init__(creator, flock_size, buff_size)
        self.dims = dims
        self.t1 = self._create_storage("t1", dims)


class DummyProc(Process):
    def run(self):
        pass

    def _check_dims(self):
        pass


def init_process(indices, subflocking=True):
    proc = DummyProc(indices, subflocking)
    return proc


def test_extra_tensors():

    num_indices = 5
    indices = torch.arange(num_indices, dtype=torch.int64)
    proc = init_process(indices)

    t = torch.rand((10, 15))
    q = torch.rand((10, 15))

    proc._read_write(t)
    proc._read_write(q)

    with pytest.raises(AssertionError):
        proc._read_write(t)

    with pytest.raises(AssertionError):
        proc._read(q)


def test_extra_buffers():

    num_indices = 5
    indices = torch.arange(num_indices, dtype=torch.int64)
    proc = init_process(indices, subflocking=False)

    creator = AllocatingCreator("cpu")
    buff = DummyBuffer(creator, 10, 100, (1,1))

    proc._get_buffer(buff)

    with pytest.raises(AssertionError):
        proc._get_buffer(buff)


def test_integrate():

    flock_size = 10

    # Those indices which are subflocked and those that are not
    indices = torch.tensor([0, 1, 3, 6, 9], dtype=torch.int64)
    non_indices = torch.tensor([2, 4, 5, 7, 8], dtype=torch.int64)

    # Define tensors for the process
    read1 = torch.full((flock_size, 10), fill_value=2)
    read2 = torch.full((flock_size, 10), fill_value=3)

    rw1 = torch.full((flock_size, 10), fill_value=4)
    rw2 = torch.full((flock_size, 10), fill_value=5)

    # Subflock via the process
    proc = init_process(indices)
    sub_read1 = proc._read(read1)
    sub_read2 = proc._read(read2)

    sub_read_write1 = proc._read_write(rw1)
    sub_read_write2 = proc._read_write(rw2)

    # Modify the subtensored lot
    sub_read1.fill_(22)
    sub_read2.fill_(100)

    sub_read_write1.fill_(15)
    sub_read_write2.fill_(8)

    # Integrate
    proc.integrate()

    # Test that we have the same read_only values
    assert same(torch.full((flock_size, 10), fill_value=2), read1)
    assert same(torch.full((flock_size, 10), fill_value=3), read2)

    # Test that the rw tensors have changed and nont changed in the relevant places
    assert same(torch.full((5, 10), fill_value=4), rw1[non_indices])
    assert same(torch.full((5, 10), fill_value=15), rw1[indices])

    assert same(torch.full((5, 10), fill_value=5), rw2[non_indices])
    assert same(torch.full((5, 10), fill_value=8), rw2[indices])




