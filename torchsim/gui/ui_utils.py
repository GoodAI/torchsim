from typing import Union

import torch
from PIL import Image
from six import BytesIO
import base64 as b64


def parse_bool(value: Union[str, bool]):
    return value == 'True' or value is True


def encode_image(tensor: torch.Tensor):
    image = (tensor * 255).byte().to("cpu").numpy()
    im = Image.fromarray(image)
    buf = BytesIO()
    im.save(buf, format='PNG')
    return b64.b64encode(buf.getvalue()).decode('utf-8')
