from dataclasses import dataclass, field
from pytest import raises
from typing import List

from torchsim.core.models.expert_params import ParamsBase


@dataclass
class NameParams:
    name: str = 'Not set'


@dataclass
class DataParams(ParamsBase):
    i: int = 0
    name: NameParams = NameParams('default name')
    l_fac: List[int] = field(default_factory=list)
    name_fac: NameParams = field(default_factory=NameParams)


class TestDataClass:
    def test_factory(self):
        d1 = DataParams()
        d2 = DataParams()
        d1.name.name = "d1"
        d1.l_fac.append(2)
        d1.name_fac.name = "f1"
        assert "f1" == d1.name_fac.name
        assert "Not set" == d2.name_fac.name  # factory creates independent instances
        assert "d1" == d1.name.name
        assert "d1" == d2.name.name  # shared instance

    def test_access_to_not_existing_fields(self):
        d1 = DataParams()
        d1.i = 1
        assert 1 == d1.i
        with raises(AttributeError):
            d1.name_missing = "abc"
            assert False is hasattr(d1, 'name_missing')
