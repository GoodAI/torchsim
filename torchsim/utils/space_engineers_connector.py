import torch

import zmq
import json
import threading
import numpy as np

from enum import Enum, auto
from typing import List

from torchsim.core.actions import AgentActionsDescriptor
from torchsim.core.exceptions import NetworkProtocolException
from torchsim.utils.sample_collection_overseer import SampleCollectionOverseer


class MouseButton(Enum):
    LEFT = auto(),
    RIGHT = auto(),
    MIDDLE = auto(),


class SpaceEngineersConnectorConfig:

    def __init__(self):
        self.agent_to_task_buffer_size: int = 20
        self.task_to_agent_buffer_size: int = 20
        self.render_width: int = 64
        self.render_height: int = 64
        self.skip_frames: int = 9
        self.object_speed_multiplier: int = 1  # set to 10 and set skip_frames to 0 for task 0
        self.reward_clipping_enabled = False
        self.curriculum = [0, -1]
        self.LOCATION_SIZE: int = 2
        self.LOCATION_SIZE_ONE_HOT: int = 100
        self.TASK_CONTROL_SIZE: int = 3
        self.TASK_METADATA_SIZE: int = 5
        # data sent by the task to the agent:
        #   1. label buffer
        #   2. location (2 floats, x y)
        #   3. label target buffer
        #   4. location target
        #   5. metadata:  metadata[0]=task_id
        #                 metadata[1]=task_instance_id
        #                 metadata[2]=task_status
        #                 metadata[3]=task_instance_status
        #                 metadata[4]=testing_phase
        #
        # data sent by the agent to the task:
        #   1. actions
        #   2. label buffer
        #   3. task control (first float - task control on/off, second float - id of task to reset and switch to,
        #                    third float - force a switch to testing phase)


class SpaceEngineersConnector:
    _actions_descriptor: AgentActionsDescriptor

    def __init__(self, ip_address, port, actions_descriptor: AgentActionsDescriptor,
                 se_config: SpaceEngineersConnectorConfig, sample_collection_overseer: SampleCollectionOverseer = None):
        # socket options
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f'tcp://{ip_address}:{port}')
        self.lock = threading.Lock()

        # polling for any listening server
        self.poller_out = zmq.Poller()
        self.poller_out.register(self.socket, zmq.POLLOUT)

        # polling for any messages from server
        self.poller_in = zmq.Poller()
        self.poller_in.register(self.socket, zmq.POLLIN)

        self._actions_descriptor = actions_descriptor
        self._se_config = se_config
        self._config_sent = False

        self._overseer = sample_collection_overseer

    def send_config_message_and_wait_for_server(self):
        # check for listening server
        while True:
            while True:
                socks_out = dict(self.poller_out.poll(100))
                if socks_out and socks_out.get(self.socket) == zmq.POLLOUT:
                    break

            self.send_config()
            try:
                # self.receive_config_confirmation()
                # instead of receiving a config confirmation, the server will send task data
                self._receive_data_from_server()  # the first data sent by the server is nonsense, we can skip it
                return
            except NetworkProtocolException as e:
                pass

    def send_config(self):
        # hack: speed up task 0 or its shortened version, task 2000 (affects accumulation of rewards)
        if len(self._se_config.curriculum) == 2 and \
                (self._se_config.curriculum[0] == 0 or self._se_config.curriculum[0] == 2000) and \
                self._se_config.curriculum[1] == -1:
            self._se_config.skip_frames = 0
            self._se_config.object_speed_multiplier = 10
            print("Modifying task 0 - skip frames = 0, object speed multiplier = 10")

        # send init message
        config = {
            "agentToTaskBufferSize": self._se_config.agent_to_task_buffer_size,
            "taskToAgentBufferSize": self._se_config.task_to_agent_buffer_size,
            "render": {
                "width": self._se_config.render_width,
                "height": self._se_config.render_height,
                "tilesInFov": 3,  # visible tiles on x axis
                "cameraAltitude": 10,  # camera altitude in meters
                "projection": "orthographic"  # orthographic|perspective
            },
            "skipFrames": self._se_config.skip_frames,
            "objectSpeedMultiplier": self._se_config.object_speed_multiplier,
            "curriculum": self._se_config.curriculum
        }

        output_message = [bytearray([1]), str.encode(json.dumps(config))]

        with self.lock:
            self.socket.send_multipart(output_message, flags=0, copy=True, track=False)
            # print("Ping server {0}, {1} bytes".format(len(output_message[0]), len(output_message[1])))

    def receive_config_confirmation(self):
        with self.lock:
            msg = self.socket.recv_multipart()
            if len(msg) > 1:
                raise NetworkProtocolException("Invalid communication: Message too long")
            header = [x for x in msg[0]][0]
            if header != 1:
                raise NetworkProtocolException("Invalid communication: Expected 1 but received {0}".format(msg[0]))

            # print("Connected, got {0} bytes".format(len(msg)))

    def send_input_receive_output(self, input: np.array, agent_to_task_label: np.array, task_control: np.array) -> \
            (np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        if not self._config_sent:  # lazy config - send when needed
            self.send_config_message_and_wait_for_server()
            self._config_sent = True

        parsed_actions = self._actions_descriptor.parse_actions(input)
        actions_bitmap = self.list_of_bools_to_bitmap(parsed_actions)
        self._send_data_to_server(actions_bitmap, agent_to_task_label, task_control)
        return self._receive_data_from_server()

    def _send_data_to_server(self, actions_bitmap, agent_to_task_label, task_control):
        output_message = [
            bytearray([2]),
            (actions_bitmap.to_bytes(4, 'little')),  # Actions bitmap
            (agent_to_task_label.astype(np.float32).tobytes()),  # agent_to_task label as floats
            (task_control.astype(np.float32).tobytes())  # task switch and reset request
        ]
        with self.lock:
            self.socket.send_multipart(output_message, flags=0, copy=True, track=False)
            # self.wait_response = True
            # print("Sent data {0}, {1} bytes".format(len(output_message[0]), len(output_message[1])))

    def _receive_data_from_server(self):
        with self.lock:
            header, image, reward, label, location, label_target, location_target, metadata = \
                self.socket.recv_multipart()
            # print("Data received {0}, {1} bytes".format(len(image), len(buffer)))

        image = np.frombuffer(image, dtype=np.uint8)
        image = image.reshape((self._se_config.render_height, self._se_config.render_width, 4))
        image = image[:, :, 0:3]

        reward = np.frombuffer(reward, dtype=np.float32, count=1)[0]
        # tta = task_to_agent
        tta_label = np.frombuffer(label, dtype=np.float32, count=self._se_config.task_to_agent_buffer_size)
        tta_location = np.frombuffer(location, dtype=np.float32, count=self._se_config.LOCATION_SIZE)
        tta_label_target = np.frombuffer(label_target, dtype=np.float32, count=self._se_config.task_to_agent_buffer_size)
        tta_location_target = np.frombuffer(location_target, dtype=np.float32,
                                            count=self._se_config.LOCATION_SIZE)
        tta_metadata = np.frombuffer(metadata, dtype=np.float32, count=self._se_config.TASK_METADATA_SIZE)

        if self._overseer is not None:
            collected_image = torch.from_numpy(image).type(torch.uint8).unsqueeze(0)
            collected_label = torch.from_numpy(tta_label).type(torch.uint8).unsqueeze(0)
            self._overseer.add_sample(collected_image, collected_label, tta_metadata[1])

        # print(f'image: {image}')
        # print(f'reward: {reward}')
        # print(f'task_to_agent_data: {task_to_agent_data}')
        image = image / 255

        return image, reward, tta_label, tta_location, tta_label_target, tta_location_target, tta_metadata

    @staticmethod
    def list_of_bools_to_bitmap(list: List[bool]) -> int:
        result = 0
        mask = 1
        for val in list:
            if val:
                result += mask
            mask *= 2
        return result
