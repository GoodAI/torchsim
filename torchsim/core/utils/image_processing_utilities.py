import torch


class ImageProcessingUtilities:

    @staticmethod
    def rgb_to_grayscale(tensor: torch.Tensor, squeeze_channel: bool):
        """conversion to luminance from RGB

         https://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color """
        (r, g, b) = torch.chunk(tensor, 3, -1)  # chunk to 3 channels along the last dimension
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if squeeze_channel:
            luminance = luminance.squeeze(-1)  # remove the last dimension, which is just a 1

        return luminance
