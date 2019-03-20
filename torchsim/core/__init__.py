import os
import sys

import numpy
import torch

FLOAT_TYPE_CUDA = torch.float32
FLOAT_TYPE_CPU = torch.float32
FLOAT_TYPE_NUMPY = numpy.float32

# For use with half precision floats, 1e-4 would be too small
if FLOAT_TYPE_CUDA == torch.float16:
    SMALL_CONSTANT = 1e-3
else:
    SMALL_CONSTANT = 1e-4

FLOAT_NAN = float('nan')
FLOAT_INF = float('inf')
FLOAT_NEG_INF = float('-inf')

on_windows = sys.platform == "Windows" or sys.platform == "win32"
SHARED_DRIVE = '\\\\goodai.com/GoodAI/' if on_windows else os.path.expanduser("~/goodai.com/")


def get_float(device) -> torch.dtype:
    if device == 'cuda':
        return FLOAT_TYPE_CUDA
    return FLOAT_TYPE_CPU
