from pathlib import Path

from torchsim.utils.os_utils import project_root_dir


def test_project_root_dir():
    path = Path(project_root_dir())
    main = path / 'main.py'
    torchsim = path / 'torchsim'
    tests = path / 'tests'
    assert True is main.exists()
    assert True is main.is_file()
    assert True is torchsim.exists()
    assert True is torchsim.is_dir()
    assert True is tests.exists()
    assert True is tests.is_dir()
