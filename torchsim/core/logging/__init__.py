import logging
import os
import sys
from typing import Optional


def _get_file_handler(filename):
    return logging.FileHandler(filename, mode='w')


def _get_console_handler():
    return logging.StreamHandler(sys.stdout)


_default_log_file = 'main.log'


def setup_logging_no_ui(filename: Optional[str] = None):
    filename = filename or _default_log_file
    _setup_logging([_get_file_handler(filename), _get_console_handler()])


class LoggerNameFormatter(logging.Formatter):
    def format(self, record):
        record.loggername = record.name.split('.')[-1]
        return super().format(record)


def _setup_logging(handlers):
    logger = logging.getLogger()
    for f in list(logger.filters):
        logger.removeFilter(f)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = LoggerNameFormatter('[%(thread)d] %(asctime)s - %(levelname).4s - %(loggername)s - %(message)s')

    # Change this when it makes sense, now we need all logging to be visible.
    if os.environ.get('LOG_DEBUG', False):
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logger.setLevel(log_level)

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def flush_logs():
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.flush()
