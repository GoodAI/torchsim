import pytest
import torch

from torchsim.core import get_float
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.expert_params import ExpertParams, NUMBER_OF_CONTEXT_TYPES
from torchsim.core.models.flock.expert_flock import ExpertFlock
from torchsim.core.utils.tensor_utils import same


def create_flock(flock_size=2,
                 input_size=7,
                 context_size=1,
                 n_providers=1,
                 device='cuda'):

    params = ExpertParams()
    params.flock_size = flock_size
    params.n_cluster_centers = 5

    sp_pooler = params.spatial
    sp_pooler.input_size = input_size
    sp_pooler.buffer_size = 10
    sp_pooler.batch_size = 3
    sp_pooler.learning_rate = 0.1
    sp_pooler.cluster_boost_threshold = 1
    sp_pooler.max_boost_time = 2
    sp_pooler.learning_period = 1

    tp_pooler = params.temporal
    tp_pooler.incoming_context_size = context_size
    tp_pooler.buffer_size = 15
    tp_pooler.batch_size = 10
    tp_pooler.learning_period = 1
    tp_pooler.seq_length = 4
    tp_pooler.seq_lookahead = 2
    tp_pooler.n_frequent_seqs = 20
    tp_pooler.max_encountered_seqs = 120
    tp_pooler.forgetting_limit = 5
    tp_pooler.n_providers = n_providers

    return ExpertFlock(params, AllocatingCreator(device))


@pytest.mark.skip(reason="Ignoring tests on non-default streams for now.")
def test_whole_flock_default_vs_nondefault_stream():
    flock_size = 1
    input_size = 7
    device = 'cuda'
    float_dtype = get_float(device)

    iterations = 50  # Needs to be high enough to run also the learning of TP.

    data = torch.rand(iterations, flock_size, input_size, dtype=float_dtype, device=device)

    flock1 = create_flock(flock_size=flock_size,
                          input_size=input_size,
                          device=device)

    flock2 = create_flock(flock_size=flock_size,
                          input_size=input_size,
                          device=device)

    flock1.copy_to(flock2)

    torch.cuda.synchronize()

    for i in range(iterations):
        flock1.run(data[i])

    torch.cuda.synchronize()

    with torch.cuda.stream(torch.cuda.Stream()):
        for i in range(iterations):
            flock2.run(data[i])

    torch.cuda.synchronize()

    assert same(flock1.tp_flock.projection_outputs, flock2.tp_flock.projection_outputs)
    assert same(flock1.tp_flock.action_rewards, flock2.tp_flock.action_rewards)
    assert same(flock1.sp_flock.cluster_centers, flock2.sp_flock.cluster_centers)


@pytest.mark.slow
@pytest.mark.parametrize('flock_size', [1, 9])
@pytest.mark.parametrize('n_providers', [1, 4])
def test_whole_flock_flock_sizes(flock_size, n_providers):
    input_size = 1
    context_size = 5
    device = 'cuda'
    float_dtype = get_float(device)

    iterations = 10  # Needs to be high enough to run also the learning of TP.

    flock = create_flock(flock_size=flock_size,
                         input_size=input_size,
                         context_size=context_size,
                         n_providers=n_providers,
                         device=device)

    for i in range(iterations):
        data = torch.rand(flock_size, input_size, dtype=float_dtype, device=device)
        context = torch.rand(flock_size, n_providers, NUMBER_OF_CONTEXT_TYPES, context_size, dtype=float_dtype, device=device)
        flock.run(data, context)
