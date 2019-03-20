from typing import List, Dict, Optional, Any


class XmlElement:
    """An XML element inside a :Document:."""
    _text: str
    _name: str
    _attributes: Dict
    _elements: List["XmlElement"]

    def __init__(self, name: str, attributes: Optional[Dict[str, str]] = None, text: str = ""):
        self._name = name
        self._attributes = attributes
        self._elements = []
        self._text = text

    def __iter__(self):
        return iter(self._elements)

    def begin(self):
        """Starts the XML element."""
        if self._attributes is None:
            return f"<{self._name}>"
        else:
            return f"<{self._name} {self.attributes_as_text}>"

    def end(self):
        """Closes the XML element."""
        return f"</{self._name}>"

    def add(self, element) -> "XmlElement":
        """Adds an XML element or string inside this element."""
        self._elements.append(element)
        return self

    def text(self):
        """Returns the text inside the XML element."""
        return self._text

    @property
    def attributes_as_text(self):
        return " ".join([f'{key}="{value}"' for key, value in self._attributes.items()])

    def generate_strings(self):
        """Generates the strings that comprise this element."""
        yield self.begin()
        if self.text():
            yield self.text()
        for e in self:
            if type(e) is str:
                yield e
            else:
                for s in e.generate_strings():
                    yield s
        yield self.end()

    def as_text(self):
        """Returns the text representing this XML element and the elements it contains."""
        return '\n'.join(self.generate_strings())
