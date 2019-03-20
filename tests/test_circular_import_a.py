# import tests.test_circular_import_b
from tests.test_circular_import_b import Circular_B


class Circular_A:
    def is_b(self, obj):
        return isinstance(obj, Circular_B)
        # return isinstance(obj, tests.test_circular_import_b.Circular_B)
