from torchsim.core.datasets.alphabet.alphabet import AlphabetGenerator


class TestAlphabetGenerator:
    def test_create_symbols(self):
        generator = AlphabetGenerator()
        result = generator.create_symbols('A:"')
        assert [3, 7, 5] == list(result.shape)
        # Check that result is composed just of zeros and ones
        zeros = (result == 0).nonzero().shape[0]
        ones = (result == 1).nonzero().shape[0]
        assert result.numel() == zeros + ones
