from torchsim.core.eval.doc_generator.element import XmlElement
from typing import List, Any, Iterable, Dict, Optional


class Document:
    """An HTML document used to record experiment output."""
    _doc_header: str
    _html: XmlElement
    _body: XmlElement
    _elements: List[XmlElement]

    def __init__(self):
        self._doc_header = "<!DOCTYPE html>"
        self._body = XmlElement('body')
        self._html = XmlElement('html')
        self._elements = [self._html.add(self._body)]

    def add(self, element: XmlElement):
        """Adds an XML element or string to the document."""
        self._body.add(element)
        return self

    def as_text(self):
        return '\n'.join([self._doc_header, self._html.as_text()])

    def write_file(self, path: str):
        with open(path, 'w') as document_file:
            document_file.write(self.as_text())

    def add_table(self, headers: Iterable[str], values: Iterable[Iterable[Any]], attribs: Optional[Dict[str, str]] = None):
        table = XmlElement('table', attribs)
        # header
        header = XmlElement('tr')
        for h in headers:
            header.add(XmlElement('th', text=h))
        table.add(header)

        # rows
        for row_values in values:
            row = XmlElement('tr')
            for cell in row_values:
                str_value = str(cell)
                row.add(XmlElement('td', {'style': 'text-align: right;'}, text=str_value))
            table.add(row)

        self.add(table)
