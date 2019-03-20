from tests.test_circular_import_a import Circular_A
from tests.test_circular_import_b import Circular_B


class TestCircularImport:
    def test_circular_import(self):
        a = Circular_A()
        b = Circular_B()
        assert True is a.is_b(b)
        assert True is b.is_a(a)
        assert False is b.is_a(b)
        assert False is a.is_b(a)
