from torchsim.core.eval.doc_generator.element import XmlElement


class Heading(XmlElement):
    _text: str

    def __init__(self, text: str, level: int = 1):
        super().__init__(f'h{level}')
        self._text = text

    def text(self):
        return self._text
