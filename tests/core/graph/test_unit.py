from tempfile import TemporaryDirectory

from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver
from torchsim.core.utils.tensor_utils import same
from tests.core.graph.node_stub import RandomUnitStub


def test_save_load():
    creator = AllocatingCreator('cpu')

    unit = RandomUnitStub(creator)
    unit2 = RandomUnitStub(creator)

    with TemporaryDirectory() as folder:
        saver = Saver(folder)
        unit.save(saver)
        saver.save()

        loader = Loader(folder)
        unit2.load(loader)

    assert same(unit.output, unit2.output)
