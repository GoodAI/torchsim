from dataclasses import dataclass
import dacite

from typing import Any, Callable, Dict, Type

import threading
import websocket
import json
import logging
import time
from websocket import WebSocketApp

from torchsim.core.exceptions import NetworkProtocolException, IllegalArgumentException
from torchsim.utils.os_utils import last_exception_as_html

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# class EventType(Enum):


@dataclass
class RequestData:
    win_id: str
    request_type: str
    data: Any


@dataclass
class EventData:
    win_id: str
    event_type: str


@dataclass
class EventDataPropertyUpdated(EventData):
    property_id: int
    value: Any


class EventParser:
    EVENT_TYPES: Dict[str, Type] = {
        'property_updated': EventDataPropertyUpdated,
        'window_closed': EventData
    }

    @classmethod
    def parse(cls, packet: Dict[str, Any]) -> EventData:
        event_type = packet['event_type']
        if event_type not in cls.EVENT_TYPES:
            raise IllegalArgumentException(f'Unrecognized event type: "{event_type}')
        t = cls.EVENT_TYPES[event_type]
        # noinspection PyTypeChecker
        return dacite.from_dict(data_class=t, data=packet)


class UIServerConnector:
    websocket: WebSocketApp = None

    def __init__(self, server: str, port: int):
        self.server = server
        self.port = port
        self.event_handlers = {}
        self.request_handlers = {}
        self.connected_to_server = False

        self.setup_socket()

    def register_event_handler(self, win_id: str, handler: Callable[[EventData], None]):
        if win_id not in self.event_handlers:
            self.event_handlers[win_id] = []
        self.event_handlers[win_id].append(handler)

    def register_request_handler(self, win_id: str, handler: Callable[[RequestData], None]):
        if win_id not in self.request_handlers:
            self.request_handlers[win_id] = []
        self.request_handlers[win_id].append(handler)

    def clear_event_handlers(self, win_id: str):
        self.event_handlers[win_id] = []

    def clear_request_handlers(self, win_id: str):
        self.request_handlers[win_id] = []

    def setup_socket(self):
        # Setup socket to server
        def on_message(_, msg):
            msg = json.loads(msg)
            command = msg['cmd']
            data = msg['data']
            # logger.info(f'Command {command}')
            # Handle server commands
            if command == 'alive_response':
                logger.info('Connected to UI server')
                self.connected_to_server = True
            elif command == 'event':
                event_data = EventParser.parse(data)
                for handler in list(self.event_handlers.get(event_data.win_id, [])):
                    # noinspection PyBroadException
                    try:
                        handler(event_data)
                    except Exception as e:
                        logger.error(last_exception_as_html())
            elif command == 'request':
                request_id = msg['requestId']
                request_data = dacite.from_dict(data_class=RequestData, data=data)

                for handler in list(self.request_handlers.get(request_data.win_id, [])):
                    result = handler(request_data)
                    self.send_request_response(request_id, result)

        def on_error(_, error):
            raise NetworkProtocolException(f"UI server connection error: {error}")

        def on_open(_):
            # Send alive confirmation request on start. Server will respond with alive_response.
            self._send_to_ui_server({
                'cmd': 'alive_request'
            })

        def on_close(_):
            self.connected_to_server = False

        def run_socket(ws_setter_: Callable[[WebSocketApp], None]):
            while True:
                try:
                    sock_addr = f"{self.server}:{self.port}/torchsim_socket"
                    ws = websocket.WebSocketApp(
                        sock_addr,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_close,
                        on_open=on_open,
                    )
                    ws_setter_(ws)
                    ws.run_forever(ping_timeout=30)
                except Exception as e:
                    logger.error(f'UI server socket error: {e}, trying to reconnect')
                time.sleep(3)

        def ws_setter(ws):
            self.websocket = ws

        # Start listening thread
        socket_thread = threading.Thread(
            target=run_socket,
            name='UI-Socket-Thread',
            args=(ws_setter,)
        )
        socket_thread.daemon = True
        socket_thread.start()

        # Block calling until the connection to UI server is established
        timeout = 5  # [s]
        time_spent = 0
        wait_time = 1
        while not self.connected_to_server and time_spent < timeout:
            time.sleep(wait_time)
            time_spent += wait_time
        if not self.connected_to_server:
            raise NetworkProtocolException(f'Could not connect to UI server')

    # Utils
    def _send_to_ui_server(self, msg):
        """Send packet to UI server"""
        self.websocket.send(json.dumps(msg))

    def send_request_response(self, request_id, data):
        return self._send_to_ui_server({
            'cmd': 'request_response',
            'requestId': request_id,
            'data': data
        })

    def send_command(self, command: str, data: Any):
        self._send_to_ui_server({
            'cmd': command,
            'data': data
        })

