import glob
import math
import os
import random
import statistics
import sys
import typing
from datetime import datetime

import torch
from genericpath import isfile
from importlib import import_module
from os.path import dirname, basename

import pytest

from eval_utils import progress_bar
from torchsim.core.exceptions import IllegalArgumentException
from torchsim.core.graph import Topology


TEMPORARY_TEST_PATH = os.path.join(os.getcwd(), 'tests', 'temp')


def get_subclasses_recursive(class_object) -> typing.Set:
    return set(class_object.__subclasses__()).union(
        [s for c in class_object.__subclasses__() for s in get_subclasses_recursive(c)]
    )


def discover_child_classes(module_name: str, base_class, skip_classes: typing.List = None) -> typing.List:
    """Discoveres classes deriving from `base_class` which are located in `module_name`.

     Skips classes that are in the `skip_classes` list.
     """
    module = import_module(module_name)
    if module is None:
        raise IllegalArgumentException(f'Cannot import module "{module_name}"')
    if module.__file__ is None:
        raise IllegalArgumentException(f'Cannot locate "{module_name}"')
    modules = glob.glob(dirname(module.__file__) + "/*.py")

    for submodule in (basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')):
        import_module(f"{module_name}.{submodule}")

    filtred = remove_abstract_classes(get_subclasses_recursive(base_class))
    if skip_classes is not None:
        filtred = remove_skipped_classes(filtred, skip_classes)
    return filtred


def is_abstract(cls):
    """Class is abstract if it derives from ABC and has at least one abstract method."""
    if not hasattr(cls, "__abstractmethods__"):
        return False  # an ordinary class
    elif len(cls.__abstractmethods__) == 0:
        return False  # a concrete implementation of an abstract class
    else:
        return True  # an abstract class


def remove_abstract_classes(topology_classes):
    return [x for x in topology_classes if not is_abstract(x)]


def remove_skipped_classes(topology_classes, skip_classes: typing.List):
    return [x for x in topology_classes if x.__name__ not in skip_classes]


def discover_main_topology_classes(skip_classes: typing.List = None):
    topology_classes = discover_child_classes('torchsim.topologies', Topology, skip_classes)
    return [x for x in topology_classes if 'torchsim.topologies' in x.__module__]


@pytest.mark.flaky(reruns=100)
@pytest.mark.parametrize('run', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_flaky(run):
    """This tests checks that flaky does work correctly."""
    r = random.random()
    assert r < 0.1


def copy_or_fill_tensor(target: torch.Tensor, values: typing.Union[torch.Tensor, int, float]):
    if isinstance(values, torch.Tensor):
        target.copy_(values)
    else:
        target.fill_(values)


def measure_time(iterations: int, function_repetitions: int = 1):
    """
    Wrapper for measuring time of cuda functions with precision of miliseconds.

    Args:
        iterations: Number of iterations of the measurement - to lower variance.
        function_repetitions: Number of function calls inside one measurement - to enable measuring of fast functions.

    Returns:

    """

    def wrap(f):
        def wrapped_f(*args):

            final = []
            torch.cuda.synchronize()
            iterator = progress_bar(range(iterations), "Measurement progress: ", file=sys.stdout)
            for step in iterator:
                start = datetime.now()
                # repeat multiple times, because the precision on windows is just mili seconds which can be too long for
                # some simple functions
                for _ in range(function_repetitions):
                    f(*args)
                    torch.cuda.synchronize()
                final.append(datetime.now() - start)

            # remove the first measurement, it usually takes much longer due to cuda init
            del final[0]

            final_ms = [math.floor(x.total_seconds() * 1000) for x in final]

            min_ms = min(final_ms)
            max_ms = max(final_ms)
            mean_ms = int(round(statistics.mean(final_ms)))

            min_us = min_ms / 1000.0
            max_us = max_ms / 1000.0
            mean_us = mean_ms / 1000.0

            assert min_ms > 0, "There was some measurement that took less than 1 ms, which is under the resolution " \
                               "of this method. Please, increase the `function_repetitions` to run the function " \
                               "multiple times per one measurement."

            print(f"\n Measurement result on {iterations - 1} iterations (first run excluded): "
                  f"\n       {function_repetitions} calls took on average {mean_ms} (min: {min_ms}, max: {max_ms})ms")
            if function_repetitions > 1:
                print(f"\n       1 call took on avearge {mean_us} (min: {min_us}, max: {max_us})ms")

        return wrapped_f
    return wrap
