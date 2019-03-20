import pytest
import tempfile
import torch

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval.doc_generator.figure import Figure
from torchsim.core.eval.doc_generator.heading import Heading
from torchsim.core.eval.doc_generator.matrix import Matrix


def test_doc_creation():
    document = Document().\
        add(Heading("Look at his heading")). \
        add("This is a very nice document.<br><br>").\
        add(Figure.from_params('test_image.png',
                               "Behold our powerful visualization",
                               height=200,
                               width=400)).\
        add(Matrix(torch.rand((2, 3)), m_labels=['a', 'b'], n_labels=['10', '20', '30'],
                   format_func=lambda x: "%.5f" % x))

    # The following line will write to a non-temporary file.
    # Uncomment if you want to see what the html file looks like.
    # document.write_file("test_file.html")

    with tempfile.TemporaryFile() as document_file:
        doc_text = document.as_text()
        document_file.write(doc_text.encode('UTF-8'))
        document_file.seek(0)
        read_text = document_file.readline()

    assert read_text.startswith(b"<!DOCTYPE html>")
