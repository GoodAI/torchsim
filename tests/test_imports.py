from importlib import import_module


def test_main_py():
    import_module('main')

def test_main_expert_flock_profiling_py():
    import_module('main_expert_flock_profiling')


# TODO add imports of files under research_topics/*/experiments/*

