import random

import numpy as np
from typing import Optional
import logging

import torch

logger = logging.getLogger()
random_seed_generator = np.random.RandomState()


def generate_seed():
    """Generate a random seed. Used in cases when we want to initialize other seed generators randomly."""
    return random_seed_generator.randint(0, 2147483647 + 1)


def _set_cuda_seed(seed: int):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()


# noinspection PyUnresolvedReferences
def _set_cuda_deterministic():
    if torch.cuda.is_available():
        logger.warning('Setting torch.backends.cudnn.deterministic to True, this may affect the performance!')
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_global_seeds(seed: Optional[int] = None):
    """Set all global seeds (np.random, global state of GPU generator, ...).

    Args:
        seed: deterministic seed for reproducible experiments. If None, a random seed is used.
    """
    if seed is None:
        seed = generate_seed()
    else:
        # in case we set non-None seed, we want the results to be replicable
        _set_cuda_deterministic()

    # CUDA
    _set_cuda_seed(seed)

    # CPU
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_rand_generator_or_set_cuda_seed(device: 'str', seed: Optional[int] = None) -> np.random.RandomState:
    """Creates an independent instance of np.random generator and seed it if required.

    Sets the global state of CUDA generator only if model on device.

    Args:
        device: cpu/cuda
        seed: seed, if None, a random seed is set

    Returns:
        Instance of np.random.RandomState random generator which should be used in the Unit.
    """
    rand = np.random.RandomState()

    if seed is None:
        seed = generate_seed()

    # CPU
    rand.seed(seed=seed)

    # CUDA
    if device == 'cuda':
        _set_cuda_seed(seed)

    return rand

