def decorate(func):
    def func_wrapper(name):
        return f'called {func}({name})'

    return func_wrapper


def decorate_class(cls):
    class Wrapper(object):
        def __init__(self, *args):
            self.wrapped = cls(*args)

        def __getattr__(self, name):
            print(f'Getting the {name} of {self.wrapped}')
            return getattr(self.wrapped, name)

        def __str__(self):
            return self.wrapped.__str__()

    return Wrapper


def decorate_field(cls):
    def func_wrapper(*args):
        return f'field {cls} {args}'

    return func_wrapper


@decorate
def get_text(name):
    return "lorem ipsum, {0} dolor sit amet".format(name)


@decorate_class
class A:
    data: int

    def __init__(self, data):
        self.data = data

    def __str__(self) -> str:
        return f'A({self.data})'


class B(A):

    def __init__(self):
        super().__init__(3)


# def __init__(self):
    #     super.__init__(3)


print(get_text("John"))

a = A(5)
print(a)
print(a.data)

b = B()
print(b)
