import pytest
import torch

from torchsim.core.utils.tensor_utils import same
from torchsim.utils.seed_utils import set_global_seeds, generate_seed


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_global_seeds(device):
    seed = 345

    set_global_seeds(seed)
    tensor1 = torch.rand([5, 2], device=device)

    set_global_seeds(seed)
    tensor2 = torch.rand([5, 2], device=device)

    assert same(tensor1, tensor2)

    seed = None

    set_global_seeds(seed)
    tensor1 = torch.rand([5, 2], device=device)

    set_global_seeds(seed)
    tensor2 = torch.rand([5, 2], device=device)

    assert not same(tensor1, tensor2)


def test_generate_seed():

    seed1 = generate_seed()
    seed2 = generate_seed()

    assert seed1 != seed2
