import logging

from torchsim.core.logging.log_observable import LogObservable


class UILogHandler(logging.Handler):
    def __init__(self, log_observable: LogObservable, level=logging.NOTSET):
        super().__init__(level)
        self._log_observable = log_observable

    def emit(self, record):
        self._log_observable.log(self.format(record))
