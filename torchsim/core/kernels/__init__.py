import logging
import os
from torch.utils.cpp_extension import load
import torch

logger = logging.getLogger('load_kernels')


def load_kernels(folder,
                 name,
                 sources,
                 extra_cflags=None,
                 extra_cuda_cflags=None,
                 extra_ldflags=None,
                 extra_include_paths=None,
                 build_directory=None,
                 verbose=True):
    logger.info(f"Loading kernels started: {name}")

    include_paths = ['torchsim/core/kernels/']
    cuda_cflags = ["-Xptxas -dlcm=ca"]

    if extra_include_paths is not None:
        include_paths += extra_include_paths

    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags

    path = os.path.dirname(os.path.abspath(folder))

    def abspath(file):
        return os.path.join(path, file)

    sources = [abspath(source) for source in sources]

    sources.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernel_helpers.cpp'))

    result = load(name,
                  sources,
                  extra_cflags,
                  cuda_cflags,
                  extra_ldflags,
                  include_paths,
                  build_directory,
                  verbose)

    logger.info("Loading kernels finished")

    return result


_get_cuda_error_code = load_kernels(__file__, 'check_cuda_errors', ['check_cuda_errors.cpp',
                                                                    'check_cuda_errors.cu']).get_cuda_error_code


def check_cuda_errors():
    error = _get_cuda_error_code()
    torch.cuda.check_error(error)
