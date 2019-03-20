from typing import Optional
from torchsim.core.logging.ui_log_handler import UILogHandler
from torchsim.core.logging.log_observable import LogObservable

from torchsim.core.logging import _default_log_file, _get_file_handler, _get_console_handler, _setup_logging
from torchsim.gui.observer_system import ObserverSystem


def setup_logging_ui(observable: LogObservable, observer_system: ObserverSystem, filename: Optional[str] = None):
    filename = filename or _default_log_file
    file_handler = _get_file_handler(filename)
    console_handler = _get_console_handler()
    _setup_logging([UILogHandler(observable), file_handler, console_handler])
    observer_system.register_observer('Log', observable)