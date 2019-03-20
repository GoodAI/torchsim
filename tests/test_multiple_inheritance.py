from typing import List


class Base:
    list: List[int]

    def __init__(self, collector):
        collector.append("Base")
        self.list = ['base']

    def add(self, item, collector):
        collector.append("Base")
        self.list.append(item)


class A(Base):
    def __init__(self, collector):
        collector.append("A")
        super().__init__(collector)

    def add(self, item, collector):
        collector.append("A")
        super().add(f'a_{item}_A', collector)


class C(A):
    def __init__(self, collector):
        collector.append("C")
        super().__init__(collector)

    def add(self, item, collector):
        collector.append("C")
        super().add(f'c_{item}_C', collector)


class B(Base):
    def __init__(self, collector):
        collector.append("B")
        super().__init__(collector)

    def add(self, item, collector):
        collector.append("B")
        super().add(f'b_{item}_B', collector)


class D(B):
    def __init__(self, collector):
        collector.append("D")
        super().__init__(collector)

    def add(self, item, collector):
        collector.append("D")
        super().add(f'd_{item}_D', collector)


class E(C, D):
    pass


def test_base_class_method_is_called_only_once():
    call_order_init = []
    call_order = []
    e = E(call_order_init)
    e.add("X", call_order)
    assert ['base', 'b_d_a_c_X_C_A_D_B'] == e.list
    assert ['C', 'A', 'D', 'B', 'Base'] == call_order
    assert ['C', 'A', 'D', 'B', 'Base'] == call_order_init
