import pytest

from torchsim.core.models.expert_params import ExpertParams


def test_max_new_seqs_default_autocalculate():
    params = ExpertParams()

    expected_max_new_seqs = params.temporal.batch_size - (params.temporal.seq_length - 1)

    assert expected_max_new_seqs == params.temporal.max_new_seqs


@pytest.mark.parametrize("new_batch_size, new_seq_length",
                          [(400, 20), (800, 73), (10, 3), (20, 4)])
def test_max_new_seqs_modified_autocalculate(new_batch_size, new_seq_length):
    params = ExpertParams()
    params.temporal.batch_size = new_batch_size
    params.temporal.seq_length = new_seq_length

    expected_max_new_seqs = params.temporal.batch_size - (params.temporal.seq_length - 1)

    assert expected_max_new_seqs == params.temporal.max_new_seqs


@pytest.mark.parametrize("desired_max_new_seqs",
                          [20, 73, 3, 4, 100, 34])
def test_max_new_seqs_overridden(desired_max_new_seqs):
    params = ExpertParams()

    params.temporal.max_new_seqs = desired_max_new_seqs

    assert desired_max_new_seqs == params.temporal.max_new_seqs
