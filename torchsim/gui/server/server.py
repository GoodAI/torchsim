import sys

import os

import json
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.escape
from abc import abstractmethod, ABC
from typing import Dict, Any, Optional
import logging

from torchsim.utils.os_utils import project_root_dir

logger = logging.getLogger(__name__)
static_path = os.path.join(project_root_dir(), 'js', 'build')


def setup_logging():
    formatter = logging.Formatter('%(asctime)s - %(levelname).4s - %(module)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class SocketSender(ABC):
    @abstractmethod
    def send(self, data):
        pass


class DataStore:
    data: Dict[str, Dict[str, Any]]
    positions: Dict[str, Dict[str, Any]]
    ui_sender: Optional[SocketSender] = None
    torchsim_sender: Optional[SocketSender] = None

    def __init__(self):
        self.data = {}
        self.positions = {}

    def put_window(self, win: str, data):
        # print(f'Add window: {win}')
        self.data[win] = data
        # self.ui_sender.send(data)

    def put_window_position(self, win: str, data):
        # print(f'Add window position: {win}')
        self.positions[win] = data

    def set_ui_socket(self, sender: SocketSender):
        self.ui_sender = sender

    def set_torchsim_socket(self, sender: SocketSender):
        self.torchsim_sender = sender

    def torchsim_message_received(self, msg):
        cmd = msg['cmd']
        if cmd == 'window':
            win_id = msg['data']['id']
            self.put_window(win_id, msg['data'])
            if win_id in self.positions:
                msg['data']['position'] = self.positions[win_id]
            self.send_to_ui(msg)
        elif cmd == 'alive_request':
            logger.info(f'TorchSim connected')
            self.torchsim_sender.send({'cmd': 'alive_response', 'data': None})
        elif cmd == 'remove_all_windows':
            for win_id in self.data:
                self.send_to_ui({
                    'cmd': 'close',
                    'data': {'id': win_id}
                })
            self.data.clear()
        else:
            self.send_to_ui(msg)
            # self.positions.clear()

    def send_to_ui(self, msg):
        if self.ui_sender is not None:
            self.ui_sender.send(msg)

    def ui_message_received(self, msg):
        # print(f'Message from UI: {msg}')
        # self.ui_sender.send(msg)
        cmd = msg['cmd']
        if cmd == 'update_window_position':
            self.put_window_position(msg['data']['id'], msg['data'])
        elif cmd == 'event':
            self.torchsim_sender.send(msg)
        elif cmd == 'request':
            self.torchsim_sender.send(msg)

    def ui_connected(self):
        for win_id, data in self.data.items():
            if win_id in self.positions:
                data['position'] = self.positions[win_id]
            self.ui_sender.send({
                'cmd': 'window',
                'data': data
            })

    def torchsim_connected(self):
        pass


class UIWebSocket(tornado.websocket.WebSocketHandler, SocketSender):
    data: DataStore

    # noinspection PyMethodOverriding
    def initialize(self, data: DataStore):
        self.data = data
        data.set_ui_socket(self)

    def open(self):
        logger.info("UI WebSocket opened")
        self.data.ui_connected()

    def on_message(self, message):
        msg = tornado.escape.json_decode(tornado.escape.to_basestring(message))
        self.data.ui_message_received(msg)

    def on_close(self):
        logger.info("UI WebSocket closed")

    def send(self, data):
        try:
            self.write_message(json.dumps(data))
        except tornado.websocket.WebSocketError as e:
            logger.error(f'Send packed failed: {e}')
        # logger.info(f"UI data sent: {data}")


class TorchSimWebSocket(tornado.websocket.WebSocketHandler, SocketSender):
    data: DataStore

    # noinspection PyMethodOverriding
    def initialize(self, data: DataStore):
        logger.info("TorchSim WebSocket initialized")
        self.data = data
        data.set_torchsim_socket(self)

    def open(self):
        logger.info("TorchSim WebSocket opened")
        self.data.torchsim_connected()

    def on_message(self, message):
        msg = tornado.escape.json_decode(tornado.escape.to_basestring(message))
        self.data.torchsim_message_received(msg)

    def on_close(self):
        logger.info("WebSocket closed")

    def send(self, data):
        try:
            self.write_message(json.dumps(data))
        except tornado.websocket.WebSocketError as e:
            logger.error(f'Send packed failed: {e}')
        # print(f"UI data sent: {data}")


class RootHandler(tornado.web.RequestHandler):
    """Server of index.html on root URL"""

    def get(self):
        self.render(os.path.join(static_path, 'index.html'))


def make_app():
    data = DataStore()
    return tornado.web.Application([
        (r"/", RootHandler),
        (r'/static/(.*)', tornado.web.StaticFileHandler, {'path': static_path}),
        (r"/socket", UIWebSocket, {'data': data}),
        (r"/torchsim_socket", TorchSimWebSocket, {'data': data}),
    ])


def run_ui_server(port: int = 5000):
    setup_logging()
    logger.info("Server started")
    app = make_app()
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    run_ui_server()
