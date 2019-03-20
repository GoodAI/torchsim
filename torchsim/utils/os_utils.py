import os
import tempfile
import traceback
from pathlib import Path


def create_dir(path):
    if os.path.isdir(path):
        return

    os.makedirs(path)


def create_temp_dir(subdir: str):
    new_dir = os.path.join(tempfile.gettempdir(), subdir)
    create_dir(new_dir)
    return new_dir


def last_exception_as_html():
    message = traceback.format_exc()
    html = f"""
        <div class="alert alert-danger">
                <h4>Exception thrown:</h4>
            <pre>{message}</pre>
        </div>
    """
    return html


def project_root_dir() -> str:
    return str(Path(__file__).parent.parent.parent.absolute())
