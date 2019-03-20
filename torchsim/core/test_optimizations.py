import os
import sys


def small_dataset_for_tests_allowed():
    use_small_datasets_string = 'USE_SMALL_DATASETS'
    if use_small_datasets_string not in os.environ:
        return False

    if os.environ[use_small_datasets_string] and 'pytest' not in sys.modules.keys():
        raise AssertionError('USE_SMALL_DATASETS is set in non-testing setting')

    return True if os.environ[use_small_datasets_string] != '0' else False
