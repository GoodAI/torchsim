# class ScopedProvider:
from typing import Callable, Dict, Any, Generic, TypeVar

from torchsim.core.exceptions import IllegalArgumentException

TKey = TypeVar('TKey')


class ResettableCache(Generic[TKey]):
    """Cache memoizing values until reset method is called."""
    _data_providers: Dict[TKey, Any]
    _stored_values: Dict[TKey, Any]

    def __init__(self):
        self._stored_values = {}
        self._data_providers = {}

    def reset(self):
        """Reset all stored values"""
        self._stored_values.clear()

    def add(self, key: TKey, data_provider: Callable[[], Any]):
        """Register data provider for the key"""
        self._data_providers[key] = data_provider

    def get(self, key: TKey):
        """Get value for the key. Cached value is returned when available (and data_provider is not called)"""
        if key in self._stored_values:
            return self._stored_values[key]
        else:
            if key not in self._data_providers:
                raise IllegalArgumentException(f'Data provider for key {key} not found')
            value = self._data_providers[key]()
            self._stored_values[key] = value
            return value


T = TypeVar('T')


class SimpleResettableCache:
    """Cache memoizing values until reset method is called."""

    _cache: ResettableCache[int]
    _key_counter: int

    def __init__(self):
        self._cache = ResettableCache()
        self._key_counter = 0

    def reset(self):
        """Reset all stored values"""
        self._cache.reset()

    def create(self, data_provider: Callable[[], T]) -> Callable[[], T]:
        """Register data_provider
        Returns getter calling data_provider when necessary - i.e. when computed value was cleared using reset()
        """
        key = self._key_counter
        self._key_counter += 1
        self._cache.add(key, data_provider)
        return lambda: self._cache.get(key)
