from torchsim.core.datasets.mnist import DatasetMNIST
from torchsim.core.test_optimizations import small_dataset_for_tests_allowed


def test_load_all():
    data, labels = DatasetMNIST().get_all()
    s = data.size()
    # self.assertEqual(data.size(), torch.Size((60000, 28, 28)))
    # self.assertEqual(labels.size(), torch.Size((60000, 28, 28)))
    if small_dataset_for_tests_allowed():
        assert data.size() == (2000, 28, 28)
        assert labels.size() == (2000,)
    else:
        assert data.size() == (60000, 28, 28)
        assert labels.size() == (60000,)


def test_get_filtered_0():
    data, labels = DatasetMNIST().get_filtered([0])
    if small_dataset_for_tests_allowed():
        assert data.size() == (191, 28, 28)
        assert labels.size() == (191,)
    else:
        assert data.size() == (5923, 28, 28)
        assert labels.size() == (5923,)


def test_get_filtered_sum_all():
    count = sum([DatasetMNIST().get_filtered([i]).labels.size(0) for i in range(10)])
    if small_dataset_for_tests_allowed():
        assert 2000 == count
    else:
        assert 60000 == count


def test_get_filtered_sum_first_three():
    count_0_1_2 = sum([DatasetMNIST().get_filtered([i]).labels.size(0) for i in range(3)])
    count_012 = DatasetMNIST().get_filtered([0, 1, 2]).labels.size(0)
    assert count_012 == count_0_1_2


