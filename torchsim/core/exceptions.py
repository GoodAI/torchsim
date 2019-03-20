from typing import TypeVar, Generic


class NetworkProtocolException(Exception):
    pass


class IllegalArgumentException(Exception):
    pass


class FailedValidationException(IllegalArgumentException):
    pass


class IllegalStateException(Exception):
    pass


class ShouldNotBeCalledException(Exception):
    pass


class TensorNotSetException(Exception):
    pass


T = TypeVar('T')


class PrivateConstructorException(Exception, Generic[T]):
    instance: T

    def __init__(self, instance: T):
        self.instance = instance
