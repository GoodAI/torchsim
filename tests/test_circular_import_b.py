import tests.test_circular_import_a


class Circular_B:
    def is_a(self, obj):
        return isinstance(obj, tests.test_circular_import_a.Circular_A)
