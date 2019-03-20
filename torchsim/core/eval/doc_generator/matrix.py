from typing import Callable, List

from torchsim.core.eval.doc_generator.element import XmlElement
from torchsim.core.eval.doc_generator.figure import Row, Cell


class Matrix(XmlElement):
    """ XML element representing a 2D matrix of any content. """
    def __init__(self,
                 matrix,
                 m_labels: List[str] = None,
                 n_labels: List[str] = None,
                 caption_text: str = None,
                 format_func: Callable = None,
                 cell_padding: int = 4):
        """

        Args:
            matrix: anything indexable over two dimensions.
            m_labels: labels of rows
            n_labels: labels of columns
            caption_text: text in left upper corner (also sets width of first column)
            format_func: function which formats element of matrix to printable (html) string
            cell_padding: specifies the space between the cell wall and the cell content
        """

        super().__init__('table', {'cellpadding': str(cell_padding), 'style': "font-family:'Courier New'"})

        if len(matrix) < 1 or len(matrix[0]) < 1:
            raise ValueError("Matrix is not indexable in two dimensions.")

        if format_func is None:
            try:
                _ = float(matrix[0][0])

                def format_func(element):
                    return str(float(element))

            except ValueError:
                def format_func(element):
                    return str(element)

        if caption_text is None:
            caption_text = "&nbsp;" * len(format_func(matrix[0][0]))

        if m_labels is None:
            m_labels = range(len(matrix))
        if n_labels is None:
            n_labels = range(len(matrix[0]))

        table_row = Row()
        table_row.add(Cell().add(f"<b>{caption_text}</b>"))
        for n_label in n_labels:
            table_row.add(Cell({"align": "center", "bgcolor": "#000000"}).
                          add(f'<b style="color:White">{n_label}</b>'))
        self.add(table_row)

        for m_label, row in zip(m_labels, matrix):
            table_row = Row()
            table_row.add(Cell({"align": "center", "bgcolor": "#000000"}).
                          add(f'<b style="color:White">{m_label}</b>'))
            for cell in row:
                table_row.add(Cell({"align": "center"}).
                              add(format_func(cell)))
            self.add(table_row)
