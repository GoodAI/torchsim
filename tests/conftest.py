import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--skip-slow', action='store', default=False
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--skip-slow'):
        return

    skip_slow = pytest.mark.skip(reason='This test is slow, run the tests with --skip-slow if you want to skip it')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)


