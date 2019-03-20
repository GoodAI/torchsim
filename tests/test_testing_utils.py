from abc import abstractmethod, ABC

from tests.testing_utils import get_subclasses_recursive, remove_abstract_classes


class Base:
    pass


class Subclass(Base):
    pass


class Grandson(Subclass):
    pass


class AbstractClass(ABC, Base):
    @abstractmethod
    def abstract_method(self):
        pass


class AbstractClassChild(AbstractClass):

    def abstract_method(self):
        pass


def test_recursive_subclass_search():
    subclasses = get_subclasses_recursive(Base)
    assert Subclass in subclasses
    assert Grandson in subclasses
    assert AbstractClass in subclasses


def test_remove_abstract_class():
    subclasses = remove_abstract_classes([Base, Subclass, Grandson, AbstractClass, AbstractClassChild])
    assert Base in subclasses
    assert Subclass in subclasses
    assert Grandson in subclasses
    assert AbstractClass not in subclasses
    assert AbstractClassChild in subclasses
