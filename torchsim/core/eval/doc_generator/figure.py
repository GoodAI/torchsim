from torchsim.core.eval.doc_generator.element import XmlElement


class Row(XmlElement):
    def __init__(self, attr=None):
        super().__init__('tr', attr)


class Cell(XmlElement):
    def __init__(self, attr=None):
        super().__init__('td', attr)


class Image(XmlElement):
    def __init__(self, path, height: int = None, width: int = None):
        attributes = {'src': path}
        if height is not None:
            attributes['height'] = f"{height}"
        if width is not None:
            attributes['width'] = f"{width}"
        super().__init__('img', attributes)


class Caption(XmlElement):
    _text: str

    def __init__(self, text: str):
        super().__init__('caption', {'align': 'bottom'})
        self._text = text

    def text(self):
        return self._text


class Figure(XmlElement):
    """XML element representing an image with a caption.

    The element is a table with a single cell. The table caption is used as the caption for the figure.
    """
    def __init__(self, image: Image, caption: Caption):
        super().__init__('table', {'class': 'image'})
        self.add(caption)
        row = Row().add(Cell().add(image))
        self.add(row)

    @classmethod
    def from_params(cls, image_path: str, caption: str, height: int = None, width: int = None):
        return cls(Image(image_path, height, width), Caption(caption))

    @classmethod
    def from_elements(cls, image: Image, caption: Caption):
        return cls(image, caption)
