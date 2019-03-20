"""Test the generic NodeBase features here."""
import pytest
from tempfile import TemporaryDirectory

import torch

from torchsim.core.graph import Topology, AllocatingCreator
from torchsim.core.graph.node_base import NodeValidationException
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver
from torchsim.core.utils.tensor_utils import same
from tests.core.graph.node_stub import RandomNodeStub, RandomUnitStub


def import_submodules(package, recursive=True):
    """Import all submodules of a module, recursively, including subpackages.

    Args:
         package: package (name or actual module)
    """
    import importlib
    import pkgutil

    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results


def test_skip_execution():
    graph = Topology('cpu')

    node = RandomNodeStub()
    graph.add_node(node)

    graph.step()
    output_1 = torch.clone(node.outputs.output.tensor)

    graph.step()
    output_2 = torch.clone(node.outputs.output.tensor)

    node.skip_execution = True
    graph.step()
    output_3 = torch.clone(node.outputs.output.tensor)

    node.skip_execution = False
    graph.step()
    output_4 = torch.clone(node.outputs.output.tensor)

    assert not same(output_1, output_2)
    assert same(output_2, output_3)
    assert not same(output_3, output_4)


def test_save_load():
    node = RandomNodeStub()
    node2 = RandomNodeStub()

    creator = AllocatingCreator('cpu')
    node.allocate_memory_blocks(creator)
    node2.allocate_memory_blocks(creator)

    with TemporaryDirectory() as folder:
        saver = Saver(folder)
        node.save(saver)
        saver.save()

        loader = Loader(folder)
        node2.load(loader)

    assert same(node._unit.output, node2._unit.output)


def test_validation():
    graph = Topology('cpu')

    class ValidationNodeStub(WorkerNodeBase):
        def __init__(self, fails_validation):
            super().__init__()
            self.fails_validation = fails_validation

        def _create_unit(self, creator: TensorCreator) -> Unit:
            return RandomUnitStub(creator)

        def _step(self):
            pass

        def validate(self):
            if self.fails_validation:
                raise NodeValidationException('Node failed to validate')

    node = ValidationNodeStub(fails_validation=True)
    graph.add_node(node)

    with pytest.raises(NodeValidationException):
        graph.prepare()

    node.fails_validation = False

    graph.prepare()


@pytest.mark.slow
def test_no_node_base_subclass_overrides_get_observables():
    # Import all nodes
    import torchsim.core.nodes
    import_submodules(torchsim.core.nodes)
    for cls in WorkerNodeBase.__subclasses__():
        overrides = 'get_observables' in cls.__dict__.keys()
        assert overrides is False, f'Class {cls.__name__} should not override ' \
                                   f'get_observables(). Override _get_observables() ' \
                                   f'instead. '
        # if has:
        #     print(cls)
