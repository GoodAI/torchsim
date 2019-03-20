import pytest
import torch
import numpy as np
from torchsim.core import get_float
from torchsim.gui.server.ui_api import UIApi


@pytest.mark.skip('Just used for manual checks of UI')
@pytest.mark.parametrize('device', ['cpu'])
class TestUIApi:

    @pytest.fixture()
    def connector(self):
        return UIApi(server='ws://localhost', port=5000)

    def test_image(self, connector, device):
        image = torch.rand((128, 64, 3), dtype=get_float(device))
        connector.image(image, f'Image {device}')

    def test_matplot(self, connector, device):
        import matplotlib.pyplot as plt
        plt.plot(np.random.randint(-10, 10, 20))
        plt.ylabel('some numbers')
        connector.matplot(plt, win=f'Matplot {device}')

    def test_text(self, connector, device):
        connector.text("Text to be <b>displayed</b><br/>New line", win='Text')
