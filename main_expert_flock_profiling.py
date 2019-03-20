import argparse
import os

from torchsim.core import FLOAT_TYPE_CPU, FLOAT_TYPE_CUDA
from torchsim.core.utils.tensor_utils import multi_unsqueeze

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.expert_params import ExpertParams, SamplingMethod, NUMBER_OF_CONTEXT_TYPES
from torchsim.core.models.flock.expert_flock import ExpertFlock, ConvExpertFlock
import torch
import time
import numpy as np

from torchsim.core.models.flock.flock_utils import memory_report
from torchsim.utils.seed_utils import set_global_seeds


def create_flock(input_size, flock_size, incoming_context_size, seq_length=4, n_cluster_centers=20,
                 max_encountered_seqs=1000, n_frequent_seqs=500, sp_buffer_size=3000, sp_batch_size=300,
                 sp_learning_period=20, convolutional: bool = False, sp_max_boost_time=1000, sp_boost_threshold=100,
                 sampling_method: SamplingMethod = SamplingMethod.BALANCED,
                 n_context_providers=1, device='cuda'):
    params = ExpertParams()
    params.n_cluster_centers = n_cluster_centers
    params.flock_size = flock_size
    params.compute_reconstruction = True

    sp_params = params.spatial
    sp_params.input_size = input_size
    sp_params.buffer_size = sp_buffer_size
    sp_params.batch_size = sp_batch_size
    sp_params.learning_rate = 0.1
    sp_params.cluster_boost_threshold = sp_boost_threshold
    sp_params.max_boost_time = sp_max_boost_time
    sp_params.learning_period = sp_learning_period
    sp_params.sampling_method = sampling_method

    tp_params = params.temporal
    tp_params.buffer_size = 100
    tp_params.seq_length = seq_length
    tp_params.batch_size = 50+tp_params.seq_length-1
    tp_params.learning_period = 50
    tp_params.seq_lookahead = 2
    tp_params.n_frequent_seqs = n_frequent_seqs
    tp_params.max_encountered_seqs = max_encountered_seqs
    tp_params.forgetting_limit = 5000
    tp_params.incoming_context_size = incoming_context_size
    tp_params.n_providers = n_context_providers

    if convolutional:
        print("Created convolutional flock.")
        return ConvExpertFlock(params, AllocatingCreator(device))
    else:
        print("Created normal flock.")
        return ExpertFlock(params, AllocatingCreator(device))


# Currently not used but will be used in the future, so should stay here

def cpu_copy_benchmark():
    input_size = 100  # decreased from 180 to fit onto a 750 Ti
    flock_size = 100

    preparation_iterations = 5
    measurement_iterations = 25
    total_iterations = preparation_iterations + measurement_iterations

    n_flocks = 5

    # flock2 = create_flock(input_size=input_size, flock_size=flock_size, seq_length=4,
    #                       context_size=input_size)
    tmem = [torch.cuda.memory_allocated()]
    allmem = [torch.cuda.max_memory_allocated()]

    print(f"\t1 flock alloc heap allocated: {allmem[0]/(1024 ** 2)}MB")
    print(f"\t1 flock alloc heap used: {tmem[0]/(1024 ** 2)}MB")

    storages = []
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    for f in range(n_flocks):
        new_flock = create_flock(input_size=input_size, flock_size=flock_size, seq_length=4,
                                 incoming_context_size=input_size)
        new_flock.to_("cpu")
        new_flock.pin()
        storages.append(new_flock)

    flock1 = create_flock(input_size=input_size, flock_size=flock_size, seq_length=4,
                          incoming_context_size=input_size)

    flock2 = create_flock(input_size=input_size, flock_size=flock_size, seq_length=4,
                          incoming_context_size=input_size)

    tmem = [torch.cuda.memory_allocated()]
    allmem = [torch.cuda.max_memory_allocated()]

    print(f"\tPost flock alloc heap allocated: {allmem[0]/(1024 ** 2)}MB")
    # print(f"\tPost flock alloc heap currently used: {tmem[0]/(1024 ** 2)}MB")

    data = torch.rand((total_iterations, flock_size, input_size), device="cuda", dtype=FLOAT_TYPE_CUDA)
    memory_for_data = (data.numel() * data.element_size()) / (1024 ** 2)

    print(f"\tMemory allocated for input data: {memory_for_data}MB")

    fwd_times = []
    learn_times = []

    # profiling on CPU
    # data = data.to("cpu")
    # flock1.to_("cpu")
    # flock2.to("cpu")

    storages[0].copy_to(flock2)
    storages[-1].copy_to(flock1)
    torch.cuda.synchronize()

    for k in range(total_iterations):
        if k % 10 == 0:
            print(f"{k} / {total_iterations}")

        if k == preparation_iterations:
            total_start = time.time()
            print("Starting timer")

        d = data[k % total_iterations]
        torch.cuda.synchronize()

        for f in range(n_flocks):
            # start copying back the result of computation of f-1

            if k % 2 == 0:
                prepared_flock = flock1
                computed_flock = flock2
            else:
                prepared_flock = flock2
                computed_flock = flock1

            finished_storage = (f - 1) % n_flocks
            prepared_storage = (f + 1) % n_flocks

            # start copying of the previous results back and the preparing new ones
            with torch.cuda.stream(copy_stream):
                prepared_flock.copy_to(storages[finished_storage])
                storages[prepared_storage].copy_to(prepared_flock)

            # meanwhile compute the computed_flock
            with torch.cuda.stream(compute_stream):
                # computed_flock.run_just_sp(d)
                computed_flock.run(d)

            torch.cuda.synchronize()

        torch.cuda.synchronize()

    total_end = time.time()

    tmem.append(torch.cuda.memory_allocated())
    allmem.append(torch.cuda.max_memory_allocated())

    # whole memory minus the one which is used for data

    memory_allocated = (np.max(allmem)) / (1024 ** 2) - memory_for_data

    print(f"Results:")
    # print(f"\tFwd pass times (avg/max/min (s)): {np.mean(fwd_times)}/{np.max(fwd_times)}/{np.min(fwd_times)}")
    # print(f"\tLearn times (avg/max/min) (s): {np.mean(learn_times)}/{np.max(learn_times)}/{np.min(learn_times)}")
    print(f"\tIterations per second: {measurement_iterations/(total_end - total_start)}")
    print(f"\tMax heap alloc (Peak mem usage): {memory_allocated}MB")
    # print(f"\tMax current heap used (Peak mem usage): {np.max(tmem)/(1024 ** 2)}MB")

    print(f"Total time: {total_end - total_start}")


def configure_bench_flock(input_size=180, flock_size=100, n_cluster_centers=20, device='cuda'):
    # profiling baseline:
    # n_cluster_centers = 20
    n_frequent_seqs = 500
    max_encountered_seqs = 1000
    sp_buffer_size = 3000
    sp_batch_size = 300
    sp_learning_period = 20
    n_context_providers = 4
    incoming_context_size = input_size // n_context_providers
    sp_max_boost_time = 1000
    sp_boost_threshold = 100
    sampling_method = SamplingMethod.BALANCED
    CONVOLUTIONAL = False

    # # convolutional:
    # sp_learning_period = sp_learning_period * flock_size
    # sp_max_boost_time = sp_max_boost_time * flock_size
    # sp_boost_threshold = sp_boost_threshold * flock_size
    # CONVOLUTIONAL = True

    # # retarded small experts 2500:
    # n_cluster_centers = 12
    # input_size = n_cluster_centers * 9
    # flock_size = 2500
    # n_frequent_seqs = 100
    # max_encountered_seqs = n_frequent_seqs * 2
    # sp_buffer_size = 500
    # sp_batch_size = 300
    # sp_learning_period = 20
    # context_size = input_size
    # sampling_method = SamplingMethod.BALANCED
    # CONVOLUTIONAL = False

    # one big expert
    # input_size = 10000
    # flock_size = 1
    # n_cluster_centers = 100
    # n_frequent_seqs = 5000
    # max_encountered_seqs = 10000
    # sp_buffer_size = 3000
    # sp_batch_size = 300
    # context_size = input_size

    # input_size = 1024
    # flock_size = 20
    # n_cluster_centers = 20
    # n_frequent_seqs = 500
    # max_encountered_seqs = 1000
    # sp_buffer_size = 300
    # sp_batch_size = 200
    # context_size = input_size

    # # Jardas flock
    # input_size = 32 * 32 * 3
    # flock_size = 2 * 2
    # sp_buffer_size = 600
    # sp_batch_size = 300
    # n_cluster_centers = 350
    # max_encountered_seqs = 100
    # n_frequent_seqs = 5
    # context_size = 1

    flock = create_flock(input_size=input_size, flock_size=flock_size, seq_length=4,
                         incoming_context_size=incoming_context_size, n_cluster_centers=n_cluster_centers,
                         max_encountered_seqs=max_encountered_seqs, n_frequent_seqs=n_frequent_seqs,
                         sp_buffer_size=sp_buffer_size, sp_batch_size=sp_batch_size, sp_max_boost_time=sp_max_boost_time,
                         sp_boost_threshold=sp_boost_threshold,
                         sp_learning_period=sp_learning_period, convolutional=CONVOLUTIONAL,
                         sampling_method=sampling_method,
                         n_context_providers=n_context_providers,
                         device=device)

    memory_report(flock, units="MB")

    return flock


def main_benchmarking_1flock(run_sp_only: bool = False, quick: bool = False):

    input_size = 180
    flock_size = 100

    flock = configure_bench_flock(input_size=input_size, flock_size=flock_size)
    context_size = flock.tp_flock.context_size
    n_providers = flock.n_providers

    preparation_iterations = 400 if quick else 1000
    measurement_iterations = 800 if quick else 1000
    total_iterations = preparation_iterations + measurement_iterations

    tmem = [torch.cuda.memory_allocated()]
    allmem = [torch.cuda.max_memory_allocated()]

    print(f"\tPost flock alloc heap allocated: {allmem[0]/(1024 ** 2)}MB")
    # print(f"\tPost flock alloc heap currently used: {tmem[0]/(1024 ** 2)}MB")

    probability_of_changed_data = 0.8
    data = torch.rand((flock_size, input_size), device="cuda", dtype=FLOAT_TYPE_CUDA)
    memory_for_data = (data.numel() * data.element_size()) / (1024 ** 2)

    print(f"\tMemory allocated for input data: {memory_for_data}MB")

    torch.cuda.synchronize()

    for k in range(total_iterations):
        if k % 10 == 0:
            print(f"{k} / {total_iterations}")

        if k == preparation_iterations:
            total_start = time.time()
            print("Starting timer")

        # make sure not all data change
        random_data = torch.rand((flock_size, input_size), device="cuda", dtype=FLOAT_TYPE_CUDA)
        indices_to_change = (torch.rand((flock_size), device='cpu', dtype=FLOAT_TYPE_CPU) < probability_of_changed_data).nonzero()
        indices_to_change = indices_to_change.to('cuda')
        n_indices_to_change = len(indices_to_change)
        if n_indices_to_change != 0:
            data.scatter_(dim=0, index=indices_to_change.view(n_indices_to_change, 1).expand(n_indices_to_change, input_size), src=random_data)
        torch.cuda.synchronize()

        if run_sp_only:
            flock.run_just_sp(data)
            # flock.run_just_tp(normalize_probs(data[:, :n_cluster_centers], dim=1),
            #     data.unsqueeze(1).expand(flock_size, 2, input_size))
        else:
            flock.run(data, data.clone().view(flock_size, n_providers, 1, context_size).expand(flock_size, n_providers,
                                                                                               3, context_size))

        # memory_allocated = (torch.cuda.memory_allocated()) / (1024 ** 2) - memory_for_data
        # max_memory_allocated = (torch.cuda.max_memory_allocated()) / (1024 ** 2) - memory_for_data
        # print(f"\tMax heap alloc (Peak mem usage): {memory_allocated} of {max_memory_allocated}MB")

        torch.cuda.synchronize()

    torch.cuda.synchronize()

    print(f"Mean # SP Forward calls: {flock.sp_flock.execution_counter_forward.type(torch.float32).mean().item()}")
    # print(f"{flock.sp_flock.execution_counter_forward}")
    print(f"Mean # SP Learning calls: {flock.sp_flock.execution_counter_learning.type(torch.float32).mean().item()}")
    print(f"Mean # TP Forward calls: {flock.tp_flock.execution_counter_forward.type(torch.float32).mean().item()}")
    print(f"Mean # TP Learning calls: {flock.tp_flock.execution_counter_learning.type(torch.float32).mean().item()}")
    # print(f"{flock.tp_flock.execution_counter_forward}")

    total_end = time.time()

    tmem.append(torch.cuda.memory_allocated())
    allmem.append(torch.cuda.max_memory_allocated())

    # whole memory minus the one which is used for data

    memory_allocated = (np.max(allmem)) / (1024 ** 2) - memory_for_data

    print(f"Results:")
    # print(f"\tFwd pass times (avg/max/min (s)): {np.mean(fwd_times)}/{np.max(fwd_times)}/{np.min(fwd_times)}")
    # print(f"\tLearn times (avg/max/min) (s): {np.mean(learn_times)}/{np.max(learn_times)}/{np.min(learn_times)}")
    it_per_sec = measurement_iterations/(total_end - total_start)
    print(f"\tIterations per second: {it_per_sec}")
    print(f"\tMax heap alloc (Peak mem usage): {memory_allocated}MB")
    # print(f"\tMax current heap used (Peak mem usage): {np.max(tmem)/(1024 ** 2)}MB")

    print(f"Total time: {total_end - total_start}")

    with open('bench.log', 'a') as bench_log:
        bench_log.write(f"{it_per_sec:.2f}\n")


def bench_2gpu():

    input_size = 50
    flock_size = 100
    n_cluster_centers = 50

    default_cuda_dev = 'cuda:0'
    flock0 = configure_bench_flock(input_size=input_size, flock_size=flock_size, n_cluster_centers=n_cluster_centers,
                                   device=default_cuda_dev)

    second_device = 'cuda:1'
    flock1 = configure_bench_flock(input_size=input_size, flock_size=flock_size, n_cluster_centers=n_cluster_centers,
                                   device=second_device)

    preparation_iterations = 100
    measurement_iterations = 400
    total_iterations = preparation_iterations + measurement_iterations

    tmem = [torch.cuda.memory_allocated()]
    allmem = [torch.cuda.max_memory_allocated()]

    print(f"\tPost flock alloc heap allocated: {allmem[0]/(1024 ** 2)}MB")
    # print(f"\tPost flock alloc heap currently used: {tmem[0]/(1024 ** 2)}MB")

    probability_of_changed_data = 0.8
    data = torch.rand((flock_size, input_size), device=default_cuda_dev, dtype=FLOAT_TYPE_CUDA)
    memory_for_data = (data.numel() * data.element_size()) / (1024 ** 2)

    print(f"\tMemory allocated for input data: {memory_for_data}MB")

    torch.cuda.synchronize()

    for k in range(total_iterations):
        if k % 10 == 0:
            print(f"{k} / {total_iterations}")

        if k == preparation_iterations:
            total_start = time.time()
            print("Starting timer")

        # make sure not all data change
        random_data = torch.rand((flock_size, input_size), device=default_cuda_dev, dtype=FLOAT_TYPE_CUDA)
        indices_to_change = (torch.rand(flock_size, device='cpu', dtype=FLOAT_TYPE_CPU) < probability_of_changed_data).nonzero()
        indices_to_change = indices_to_change.to(default_cuda_dev)
        n_indices_to_change = len(indices_to_change)
        if n_indices_to_change != 0:
            data.scatter_(dim=0, index=indices_to_change.view(n_indices_to_change, 1).expand(n_indices_to_change, input_size), src=random_data)
        torch.cuda.synchronize()

        flock0.run(data, data.unsqueeze(1).expand(flock_size, 2, input_size).clone())

        # data to copy in a multi_gpu test
        non_blocking = True
        flock0.sp_flock.current_reconstructed_input.to(device=second_device, non_blocking=non_blocking)
        flock0.sp_flock.predicted_reconstructed_input.to(device=second_device, non_blocking=non_blocking)
        flock0.sp_flock.forward_clusters.to(device=second_device, non_blocking=non_blocking)

        projection_output = flock0.tp_flock.projection_outputs.to(device=second_device, non_blocking=non_blocking)
        output_context = flock0.output_context.to(device=second_device, non_blocking=non_blocking)

        torch.cuda.synchronize()

        # TODO: better to handle the dimensions by setting different context size
        flock1.run(projection_output, output_context[:, :, input_size:])

        # memory_allocated = (torch.cuda.memory_allocated()) / (1024 ** 2) - memory_for_data
        # max_memory_allocated = (torch.cuda.max_memory_allocated()) / (1024 ** 2) - memory_for_data
        # print(f"\tMax heap alloc (Peak mem usage): {memory_allocated} of {max_memory_allocated}MB")

        torch.cuda.synchronize()

    torch.cuda.synchronize()

    print(f"Mean # SP Forward calls: {flock0.sp_flock.execution_counter_forward.type(torch.float32).mean().item()}")
    # print(f"{flock.sp_flock.execution_counter_forward}")
    print(f"Mean # SP Learning calls: {flock0.sp_flock.execution_counter_learning.type(torch.float32).mean().item()}")
    print(f"Mean # TP Forward calls: {flock0.tp_flock.execution_counter_forward.type(torch.float32).mean().item()}")
    print(f"Mean # TP Learning calls: {flock0.tp_flock.execution_counter_learning.type(torch.float32).mean().item()}")
    # print(f"{flock.tp_flock.execution_counter_forward}")

    total_end = time.time()

    tmem.append(torch.cuda.memory_allocated())
    allmem.append(torch.cuda.max_memory_allocated())

    # whole memory minus the one which is used for data

    memory_allocated = (np.max(allmem)) / (1024 ** 2) - memory_for_data

    print(f"Results:")
    # print(f"\tFwd pass times (avg/max/min (s)): {np.mean(fwd_times)}/{np.max(fwd_times)}/{np.min(fwd_times)}")
    # print(f"\tLearn times (avg/max/min) (s): {np.mean(learn_times)}/{np.max(learn_times)}/{np.min(learn_times)}")
    it_per_sec = measurement_iterations/(total_end - total_start)
    print(f"\tIterations per second: {it_per_sec}")
    print(f"\tMax heap alloc (Peak mem usage): {memory_allocated}MB")
    # print(f"\tMax current heap used (Peak mem usage): {np.max(tmem)/(1024 ** 2)}MB")

    print(f"Total time: {total_end - total_start}")

    with open('bench.log', 'a') as bench_log:
        bench_log.write(f"{it_per_sec:.2f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sp-only', action='store_true', help="Use SP only in the standard benchmark.")
    parser.add_argument('-q', '--quick', action='store_true', help="Quick. Run for fewer iterations.")
    parser.add_argument('-1', '--device1', action='store_true', help="Set CUDA device to 1.")
    parser.add_argument('bench', nargs='?', help="Which benchmark: 'flock' (default), 'cpu-copy', '2gpu'")
    args = parser.parse_args()

    set_global_seeds(100)
    torch.cuda.set_device(0 if not args.device1 else 1)

    # os.environ['THC_CACHING_ALLOCATOR'] = '0'
    torch.set_grad_enabled(False)
    # main_simple()
    # main_mnist()
    # sys.settrace(trace_memory_allocations)

    if args.bench == 'cpu-copy':
        cpu_copy_benchmark()
    elif args.bench == 'flock' or args.bench is None:
        main_benchmarking_1flock(run_sp_only=args.sp_only, quick=args.quick)
    elif args.bench == '2gpu':
        bench_2gpu()
    else:
        print(f"Unknown benchmark: '{args.bench}'\n")
        parser.print_help()
