import os
import sys
import time

import torch

from torchsim.core import FLOAT_TYPE_CPU, FLOAT_TYPE_CUDA
from torchsim.utils.seed_utils import set_global_seeds

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def bench_block_size(block_size_mb=16):

    assert FLOAT_TYPE_CUDA == torch.float32, "unexpected cuda float type"
    float_size = 4
    block_size = block_size_mb // float_size

    preparation_iterations = 10
    measurement_iterations = 2048 // block_size_mb if block_size_mb <= 128 else 10
    total_iterations = preparation_iterations + measurement_iterations

    vector_count = 3
    data = [torch.rand((block_size, 1024, 1024), device="cuda", dtype=FLOAT_TYPE_CUDA) for _ in range(vector_count)]

    for k in range(total_iterations):
        if k == preparation_iterations:
            torch.cuda.synchronize()
            total_start = time.time()
            print("Starting timer")
        # swap data
        non_blocking = True
        data[1].copy_(data[0], non_blocking=non_blocking)
        data[0].copy_(data[2], non_blocking=non_blocking)
        data[2].copy_(data[1], non_blocking=non_blocking)

    torch.cuda.synchronize()
    total_end = time.time()
    elapsed_time = total_end - total_start

    print(f"\tIterations per second: {measurement_iterations / elapsed_time:.1f}")
    print(f"\tTotal time: {elapsed_time:.3f}")

    print(f"\tAssuming float type size: {float_size} B")
    print(f"Block size: {block_size * float_size} MB")
    speed_gbps = block_size * float_size * vector_count * measurement_iterations / elapsed_time / 1024.0
    print(f"*** Copying speed ***: {speed_gbps:.1f} GB/s\n")


def bench():
    for block_size_mb in [16, 256]:
        bench_block_size(block_size_mb)


if __name__ == '__main__':
    set_global_seeds(100)
    torch.cuda.set_device(0)

    # os.environ['THC_CACHING_ALLOCATOR'] = '0'
    torch.set_grad_enabled(False)

    bench()
