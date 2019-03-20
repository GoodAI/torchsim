from PIL import Image
from PIL import ImageFont, ImageDraw
import numpy as np
import torch
import os

from torchsim.gui.validators import validate_predicate


class AlphabetGenerator:
    _padding_right: int

    def __init__(self, padding_right: int = 0):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        font_path = os.path.join(script_dir, '5x5.ttf')
        self._usr_font = ImageFont.truetype(font_path, 10)
        self._padding_right = padding_right

    def _create_symbol(self, symbol: str) -> torch.Tensor:
        """Create tensor containing rendered symbol using 5x5 font
        Args:
            symbol: Symbol to be rendered. String must be of size 1
        Returns:
            Tensor[dtype=uint8] of dimensions [7, 5 + self._padding_right] - by default [7,5] as padding_right is 0.
            Pixel values are: 1 - symbol, 0 - background
        """
        validate_predicate(lambda: len(symbol) == 1)

        image = Image.new("RGBA", (5 + self._padding_right, 10), (255, 255, 255))
        d_usr = ImageDraw.Draw(image)
        d_usr.fontmode = "1"  # turn off antialiasing
        d_usr.text((0, 0), symbol, (0, 0, 0), font=self._usr_font)

        tensor = torch.from_numpy(np.array(image))
        # [height, width, channels]
        # cut top 3 lines (they are necessary in order the font is rendered in correct size)
        # convert tensor to bitmap (dtype = uint8) of dimensions [height-3, width]
        return tensor[3:, :, 0] == 0

    def create_symbols(self, symbols: str) -> torch.Tensor:
        """Render multiple symbols using 5x5 font
        Args:
            symbols: Symbols to be rendered

        Returns:
            Tensor[dtype=uint8] of dimensions [symbol_count, 7, 5 + self._padding_right]
            Pixel values are: 1 - symbol, 0 - background
        """
        rendered_symbols = [self._create_symbol(s) for s in symbols]
        return torch.stack(rendered_symbols)
