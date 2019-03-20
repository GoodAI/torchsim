import logging

from torchsim.core.graph.slots import InputSlot, OutputSlotBase


class Connection:
    """Connects OutputSlotBase to InputSlot."""
    _source: OutputSlotBase
    _target: InputSlot
    _is_backward: bool

    def __init__(self, source_output: OutputSlotBase, target_input: InputSlot, is_backward: bool = False):
        self._source = source_output
        self._target = target_input
        self._is_backward = is_backward

    @property
    def source(self) -> OutputSlotBase:
        return self._source

    @property
    def target(self) -> InputSlot:
        return self._target

    @property
    def is_backward(self):
        return self._is_backward


class InputAlreadyUsedException(Exception):
    pass


class ConnectionNotPresentException(Exception):
    pass


class Connector:
    """Serves for connecting/disconnecting Nodes."""

    @staticmethod
    def connect(source_output: OutputSlotBase, target_input: InputSlot, is_backward: bool = False) -> Connection:
        if target_input.connection is not None:
            message = f"The input {target_input.name} of node {target_input.owner.name} is already connected"
            raise InputAlreadyUsedException(message)

        connection = Connection(source_output, target_input, is_backward)

        source_output.add_connection(connection)
        target_input.connection = connection

        return connection

    @staticmethod
    def disconnect_input(target_input: InputSlot):
        if target_input.connection is None:
            logger = logging.getLogger(f"Connection -> {target_input.owner.name}")
            logger.error("ERROR, given input not connected")
            return

        connection = target_input.connection

        connection.source.remove_connection(connection)
        connection.target.connection = None

    """Connects two OutputSlotBases"""
    _source: OutputSlotBase
    _target: InputSlot