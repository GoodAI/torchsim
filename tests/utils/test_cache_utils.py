from torchsim.utils.cache_utils import ResettableCache, SimpleResettableCache


class TestResettableCache:

    def test_basic_usage(self):
        counter = [0]

        def data_provider() -> int:
            counter[0] += 1
            return counter[0]

        scope: ResettableCache[str] = ResettableCache()
        key = 'value1'
        scope.add(key, data_provider)
        assert 1 == scope.get(key)
        assert 1 == scope.get(key)
        scope.reset()
        assert 2 == scope.get(key)
        assert 2 == scope.get(key)
        scope.reset()
        assert 3 == scope.get(key)


class TestSimpleResettableCache:
    def test_create(self):
        counter = [0]

        def data_provider() -> int:
            counter[0] += 1
            return counter[0]

        scope = SimpleResettableCache()
        getter = scope.create(data_provider)
        assert 0 == counter[0]
        assert 1 == getter()
        assert 1 == getter()
        scope.reset()
        assert 2 == getter()
        assert 2 == getter()
        scope.reset()
        assert 3 == getter()
