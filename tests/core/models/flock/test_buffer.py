import torch
import pytest

from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.flock.buffer import Buffer, CurrentValueNotStoredException

from torchsim.core import FLOAT_NAN, get_float, FLOAT_TYPE_CUDA, FLOAT_TYPE_CPU
from torchsim.core.models.spatial_pooler import SPFlock
from torchsim.core.utils.tensor_utils import same

DEVICE = 'cpu'



def create_buffer(flock_size=3, buffer_size=10, last_dim=3, device=DEVICE):
    buff = Buffer(AllocatingCreator(device), flock_size=flock_size, buffer_size=buffer_size)

    buff.t1 = buff._create_storage("t1", (flock_size, buffer_size, last_dim))
    return buff


def test_buffer_get_stored():
    buffer = create_buffer(flock_size=3, buffer_size=10, last_dim=3)
    buffer.set_flock_indices(torch.tensor([0, 2]))
    buffer.t1.stored_data.uniform_()

    expected = buffer.t1.stored_data[[0, 2]]
    actual = buffer.t1.get_stored_data()

    assert same(expected, actual)


def test_buffer_set_stored():
    buffer = create_buffer(flock_size=3, buffer_size=10, last_dim=3)
    buffer.set_flock_indices(torch.tensor([0, 2]))

    expected = torch.rand(3, 10, 3)
    expected[1] = FLOAT_NAN

    buffer.t1.set_stored_data(expected[[0, 2]])

    assert same(expected, buffer.t1.stored_data)


def test_buffer_storing():
    buffer = create_buffer(flock_size=3, buffer_size=10, last_dim=3)

    inp = torch.tensor([[0.1, 0.1, 0.1],
                        [0.2, 0.2, 0.2],
                        [0.3, 0.3, 0.3]], dtype=buffer._float_dtype, device=DEVICE)

    buffer.current_ptr = torch.tensor([0, 9, 3], dtype=torch.int64, device=DEVICE)

    buffer.t1.store(inp)

    ground_truth_inputs = torch.full((3, 10, 3), fill_value=FLOAT_NAN, device=DEVICE)
    ground_truth_inputs[0, 0] = inp[0]
    ground_truth_inputs[1, 9] = inp[1]
    ground_truth_inputs[2, 3] = inp[2]

    ground_truth_pointer = torch.tensor([0, 9, 3], dtype=torch.int64, device=DEVICE)

    assert same(buffer.t1.stored_data, ground_truth_inputs)
    assert same(buffer.current_ptr, ground_truth_pointer)


def test_buffer_mask_creation():
    buffer = create_buffer(flock_size=3, buffer_size=10, last_dim=3)
    buffer.current_ptr = torch.tensor([1, 3, 6], dtype=torch.int64, device=DEVICE)
    buffer.t1.stored_data[0][1] = torch.tensor([0.1, 0.1, 0.1], dtype=buffer._float_dtype, device=DEVICE)
    buffer.t1.stored_data[1][3] = torch.tensor([0.2, 0.2, 0.2], dtype=buffer._float_dtype, device=DEVICE)

    inp = torch.tensor([[0.1, 0.1, 0.1],
                        [0.2, 0.2, 0.2],
                        [0.3, 0.3, 0.3]], dtype=buffer._float_dtype, device=DEVICE)

    ground_truth = torch.tensor([0, 0, 1], dtype=torch.uint8, device=DEVICE)

    eq_data = buffer.t1.compare_with_last_data(inp, SPFlock._detect_any_difference)

    assert same(eq_data, ground_truth)


def test_bad_batch_size():
    buffer = create_buffer(flock_size=7, buffer_size=1000, last_dim=3)
    with pytest.raises(AssertionError):
        buffer.t1._sample_batch(1001, torch.zeros((7, 1001, 3), device=DEVICE))

    buffer.t1._sample_batch(1000, torch.zeros((7, 1000, 3), device=DEVICE))


def test_batch_sampling():
    buffer = create_buffer(flock_size=7, buffer_size=1000, last_dim=5)

    buffer.t1.stored_data = torch.randn(buffer.t1.stored_data.size(), dtype=buffer._float_dtype, device=DEVICE)
    batch_size = 100

    sample = torch.zeros((7, 100, 5), dtype=buffer._float_dtype, device=DEVICE)

    # Test 1: Past is contiguous in the buffer, pointers equal
    buffer.current_ptr = torch.full((7,), fill_value=990, dtype=torch.int64, device=DEVICE)
    buffer.t1._sample_batch(batch_size, sample)

    # Permute for batch-major format
    ground_truth_sample = buffer.t1.stored_data[:, 891:991]

    assert same(sample, ground_truth_sample)

    # Test2: Past is non-contiguous in the buffer, pointers equal
    buffer.current_ptr = torch.full((7,), fill_value=50, dtype=torch.int64, device=DEVICE)
    buffer.t1._sample_batch(batch_size, sample)

    # Permute for batch-major format
    ground_truth_sample = torch.cat([buffer.t1.stored_data[:, -49:],
                                     buffer.t1.stored_data[:, :51]],
                                    dim=1)

    assert same(sample, ground_truth_sample)

    # Test3: Past is uneven in the buffer, pointers unequal
    buffer.current_ptr = torch.tensor([50, 200, 10, 500, 70, 95, 33], dtype=torch.int64, device=DEVICE)
    ground_truth_sample = [torch.cat([buffer.t1.stored_data[0, -49:], buffer.t1.stored_data[0, :51]]),
                           buffer.t1.stored_data[1, 101:201],
                           torch.cat([buffer.t1.stored_data[2, -89:], buffer.t1.stored_data[2, :11]]),
                           buffer.t1.stored_data[3, 401:501],
                           torch.cat([buffer.t1.stored_data[4, -29:], buffer.t1.stored_data[4, :71]]),
                           torch.cat([buffer.t1.stored_data[5, -4:], buffer.t1.stored_data[5, :96]]),
                           torch.cat([buffer.t1.stored_data[6, -66:], buffer.t1.stored_data[6, :34]])]

    # After stacking, convert to batch-major via permute
    ground_truth_sample = torch.stack(ground_truth_sample)
    buffer.t1._sample_batch(batch_size, sample)

    assert same(sample, ground_truth_sample)

    # Test4: Test different batch-sizes - smaller
    batch_size = 1
    sample = torch.zeros((7, 1, 5), dtype=buffer._float_dtype, device=DEVICE)
    buffer.current_ptr = torch.full((7,), fill_value=500, dtype=torch.int64, device=DEVICE)
    buffer.t1._sample_batch(batch_size, sample)

    ground_truth_sample = buffer.t1.stored_data[:, (501 - batch_size):501]

    assert same(sample, ground_truth_sample)

    # Test5: Test different batch-sizes - bigger
    batch_size = 423
    sample = torch.zeros((7, 423, 5), dtype=buffer._float_dtype, device=DEVICE)
    buffer.current_ptr = torch.full((7,), fill_value=500, dtype=torch.int64, device=DEVICE)
    buffer.t1._sample_batch(batch_size, sample)

    ground_truth_sample = buffer.t1.stored_data[:, (501 - batch_size):501]

    assert same(sample, ground_truth_sample)


def test_buffer_index_storing():
    device = 'cuda'
    buffer = create_buffer(flock_size=10, buffer_size=10, last_dim=3, device=device)

    # Define indices that the buffer is to touch and co-define those which it's not meant to
    indices = torch.tensor([0, 3, 5, 8], dtype=torch.int64, device=device)
    non_indices = torch.tensor([1, 2, 4, 6, 7, 9], dtype=torch.int64, device=device)
    buffer.set_flock_indices(indices)

    original_t1 = torch.rand((10, 10, 3), dtype=buffer._float_dtype, device=device)
    original_new_t1_data = torch.rand((4, 3), dtype=buffer._float_dtype, device=device)

    buffer.current_ptr = torch.tensor([0, 1, 2, 4, 1, 0, 9, 9, 3, 8], dtype=torch.int64, device=device)
    buffer.t1.stored_data = original_t1.clone()

    # Store the data in the buffer
    buffer.t1.store(original_new_t1_data.clone())

    # Check that the stuff that was not meant to be touched is the same as the original values
    assert same(original_t1[non_indices], buffer.t1.stored_data[non_indices])
    # Check that the stuff that should have been touched is properly written to
    assert same(original_new_t1_data, buffer.t1.stored_data[indices, buffer.current_ptr[indices]])


def test_buffer_index_sampling():
    buffer = create_buffer(flock_size=10, buffer_size=10, last_dim=3)
    batch_size = 5

    # Define indices that the buffer is to touch
    indices = torch.tensor([0, 3, 5, 8], dtype=torch.int64, device=DEVICE)
    buffer.set_flock_indices(indices)

    original_t1 = torch.rand((10, 10, 3), device=DEVICE)
    buffer.t1.stored_data = original_t1.clone()

    batch_t = torch.zeros((4, batch_size, 3), device=DEVICE)

    buffer.current_ptr = torch.tensor([0, 1, 2, 4, 1, 2, 9, 9, 6, 8], dtype=torch.int64, device=DEVICE)
    expected_batch = [torch.cat([original_t1[0, -4:], original_t1[0, 0].unsqueeze(0)]),
                      original_t1[3, :5],
                      torch.cat([original_t1[5, -2:], original_t1[5, :3]]),
                      original_t1[8, 2:7]]

    expected_batch = torch.stack(expected_batch)
    buffer.t1._sample_batch(batch_size, batch_t)

    assert same(expected_batch, batch_t)


def test_next_step():
    device = 'cuda'
    flock_size = 5
    buffer_size = 10
    input_size = 2

    buffer = create_buffer(flock_size=flock_size, buffer_size=buffer_size, last_dim=input_size, device=device)

    buffer.current_ptr = torch.tensor([0, 2, 3, 9, 4], dtype=torch.int64, device=device)
    buffer.flock_indices = torch.tensor([1, 3, 4], dtype=torch.int64, device=device)

    with buffer.next_step():
        data = torch.tensor([[0.1, 0.1],
                             [0.5, 0.2],
                             [0.2, 0.2]], dtype=buffer._float_dtype, device=device)
        buffer.t1.store(data)

    expected_ptr = torch.tensor([0, 3, 3, 0, 5], dtype=torch.int64, device=device)

    assert same(expected_ptr, buffer.current_ptr)

    with pytest.raises(CurrentValueNotStoredException):
        with buffer.next_step():
            pass


@pytest.mark.parametrize('steps', [1, 7])
def test_next_n_steps(steps):
    device = 'cuda'
    flock_size = 5
    buffer_size = 10
    input_size = 2

    buffer = create_buffer(flock_size=flock_size, buffer_size=buffer_size, last_dim=input_size, device=device)

    buffer.current_ptr = torch.tensor([0, 2, 3, 9, 4], dtype=torch.int64, device=device)
    buffer.flock_indices = torch.tensor([1, 3, 4], dtype=torch.int64, device=device)

    with buffer.next_n_steps(steps):
        data = torch.tensor([[0.1, 0.1],
                             [0.5, 0.2],
                             [0.2, 0.2]], dtype=buffer._float_dtype, device=device)
        for step in range(steps):
            buffer.t1.store(data)

    expected_ptr = torch.tensor([0, (2 + steps) % buffer_size, 3, (9 + steps) % buffer_size, (4 + steps) % buffer_size],
                                dtype=torch.int64, device=device)

    assert same(expected_ptr, buffer.current_ptr)

    # not storing anything
    with pytest.raises(CurrentValueNotStoredException):
        with buffer.next_n_steps(steps):
            pass

    # storing too little
    with pytest.raises(CurrentValueNotStoredException):
        with buffer.next_n_steps(steps):
            for step in range(steps-1):
                buffer.t1.store(data)

    # storing too much
    with pytest.raises(CurrentValueNotStoredException):
        with buffer.next_n_steps(steps):
            for step in range(steps+1):
                buffer.t1.store(data)


def test_reorder():
    def test_reorder_with_buffer_indices(flock_size, buffer_size, input_size, buffer_indices, expected_data):
        buffer = create_buffer(flock_size=flock_size, buffer_size=buffer_size, last_dim=input_size)

        indices = torch.tensor([[3, 4, 0, 15], [3, 2, 1, 4]], dtype=torch.int64, device=DEVICE)

        buffer.t1.stored_data = torch.tensor([[[0, 1, 2, 3],
                                               [4, 5, 6, 7],
                                               [8, 9, 10, 11]],
                                              [[12, 13, 14, 15],
                                               [16, 17, 18, 19],
                                               [20, 21, 22, 23]]], dtype=buffer._float_dtype, device=DEVICE)

        original_tensor = buffer.t1.stored_data

        seq_indices = indices[buffer_indices]
        expected_seq_indices = seq_indices.clone()
        if buffer_indices.size()[0] == flock_size:
            buffer.flock_indices = None  # whole flock running
        else:
            buffer.flock_indices = buffer_indices  # subflocking

        buffer.t1.reorder(seq_indices)

        assert same(expected_data, buffer.t1.stored_data)
        assert same(expected_seq_indices, seq_indices)  # check that it did not change the seq_indices
        assert original_tensor.data_ptr() == buffer.t1.stored_data.data_ptr()  # check that it did not change the pointer to the tensor

    flock_size = 2
    buffer_size = 3
    input_size = 4

    nan = FLOAT_NAN
    float_dtype = get_float(DEVICE)

    flock_indices1 = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)

    # Test with indices == None
    expected_data1 = torch.tensor([[[3, nan, 0, nan],
                                    [7, nan, 4, nan],
                                    [11, nan, 8, nan]],
                                   [[15, 14, 13, nan],
                                    [19, 18, 17, nan],
                                    [23, 22, 21, nan]]], dtype=float_dtype, device=DEVICE)

    test_reorder_with_buffer_indices(flock_size, buffer_size, input_size, buffer_indices=flock_indices1,
                                     expected_data=expected_data1)

    # Test with only ordering one expert

    flock_indices2 = torch.tensor([1], dtype=torch.int64, device=DEVICE)

    expected_data2 = torch.tensor([[[0, 1, 2, 3],
                                    [4, 5, 6, 7],
                                    [8, 9, 10, 11]],
                                   [[15, 14, 13, nan],
                                    [19, 18, 17, nan],
                                    [23, 22, 21, nan]]], dtype=float_dtype, device=DEVICE)

    test_reorder_with_buffer_indices(flock_size, buffer_size, input_size, buffer_indices=flock_indices2,
                                     expected_data=expected_data2)


def test_buffer_batch_storing():
    buffer = create_buffer(flock_size=1, buffer_size=10, last_dim=3)

    inp = torch.tensor([[[0.1, 0.1, 0.1],
                         [0.2, 0.2, 0.2],
                         [0.3, 0.3, 0.3]]], dtype=buffer._float_dtype, device=DEVICE)

    buffer.current_ptr = torch.tensor([2], dtype=torch.int64, device=DEVICE)

    buffer.t1.store_batch(inp)

    ground_truth_inputs = torch.full((1, 10, 3), fill_value=FLOAT_NAN, device=DEVICE)
    ground_truth_inputs[0, 0] = inp[0, 0]
    ground_truth_inputs[0, 1] = inp[0, 1]
    ground_truth_inputs[0, 2] = inp[0, 2]

    ground_truth_pointer = torch.tensor([2], dtype=torch.int64, device=DEVICE)

    assert same(buffer.t1.stored_data, ground_truth_inputs)
    assert same(buffer.current_ptr, ground_truth_pointer)


@pytest.mark.parametrize("flock_indices", [([0, 1]), ([0, 1, 2]), ([0, 2]), None])
def test_buffer_force_cpu(flock_indices):
    device = 'cuda'
    flock_size = 3
    buffer_size = 10
    last_dim = 4

    # Create the buffer
    buffer = Buffer(AllocatingCreator(device), flock_size=flock_size, buffer_size=buffer_size)
    buffer.t1 = buffer._create_storage("t1", (flock_size, buffer_size, last_dim), force_cpu=True)

    # Assemble the flock indices
    if flock_indices is not None:
        flock_indices = torch.tensor(flock_indices, dtype=torch.int64)
        buffer.set_flock_indices(flock_indices)
        flock_ind_len = flock_indices.numel()
    else:
        flock_ind_len = flock_size

    storing_data = torch.ones((flock_ind_len, last_dim), dtype=FLOAT_TYPE_CUDA, device=device)
    expected_stored_data = storing_data.to('cpu').type(FLOAT_TYPE_CPU)

    buffer.t1.store(storing_data)

    if flock_indices is not None:
        actual_stored_data = buffer.t1.stored_data[flock_indices, -1]
    else:
        actual_stored_data = buffer.t1.stored_data[:, -1]

    # Test some storing!
    assert same(expected_stored_data, actual_stored_data)

    # Test comparison
    expected_comparison = torch.zeros(flock_size, dtype=torch.uint8, device=device)
    data = buffer.t1.stored_data[:, -1].type(FLOAT_TYPE_CUDA).to(device)

    actual_comparison = buffer.t1.compare_with_last_data(data, SPFlock._detect_any_difference)

    assert same(expected_comparison, actual_comparison)

    # Test_sampling

    out_tensor = torch.zeros(flock_ind_len, 3, last_dim, dtype=FLOAT_TYPE_CUDA, device=device)
    buffer.t1.sample_contiguous_batch(3, out_tensor)

    expected_out_tensor = torch.full((flock_ind_len, 3, last_dim), fill_value=FLOAT_NAN, dtype=FLOAT_TYPE_CUDA, device=device)
    expected_out_tensor[:, -1] = 1

    assert same(expected_out_tensor, out_tensor)

