import math

import dataclasses
import io
import re
from dataclasses import dataclass
from typing import List, Any, Optional, Callable

import torch
from PIL import Image
from six import BytesIO
import base64 as b64
import matplotlib.pyplot as plt

from torchsim.gui.observables import ObserverPropertiesItem
from torchsim.gui.observers.cluster_observer import ClusterObserverData
from torchsim.gui.observers.tensor_observable import TensorViewProjectionUIParams
from torchsim.gui.server.ui_server_connector import UIServerConnector, EventData, RequestData
from torchsim.gui.ui_utils import encode_image
from torchsim.utils.dict_utils import to_nested_dict


@dataclass
class MemoryBlockParams:
    scale: int
    projection: TensorViewProjectionUIParams
    current_ptr: Optional[List[int]]


class UIApi:
    def __init__(self, server: str, port: int):
        self._connector = UIServerConnector(server, port)

    @staticmethod
    def sanitize_json_numbers(values: List[List[float]]):
        def replace_value(value: float):
            if math.isnan(value):
                return 'NaN'
            elif math.isinf(value) and value > 0:
                return 'Inf'
            elif math.isinf(value) and value < 0:
                return '-Inf'
            else:
                return value

        return [[replace_value(i) for i in l] for l in values]

    # noinspection PyIncorrectDocstring
    def memory_block(self, tensor: torch.Tensor, params: MemoryBlockParams,
                     properties: List[ObserverPropertiesItem], win: str):
        """
        Create memory_block observer.

        Args:
            tensor: Expected dimensions: [height, width, RGB], values 0 - 1.0
        """
        height, width, _ = tensor.size()
        b64encoded = encode_image(tensor)

        values_params = {
            'width': width,
            'height': height,
            'data': self.sanitize_json_numbers([[0]])
        }

        content = {
            'src': 'data:image/png;base64,' + b64encoded,
            'values': values_params,
            'params': dataclasses.asdict(params),
            'properties': properties
        }
        self._send_command_window(win, 'memory_block', content)

    def hierarchical_observer(self, groups_stacking, items_per_row: int, image_groups: List[List[torch.Tensor]],
                              params: List[dict], properties: List[ObserverPropertiesItem], win: str):
        """This function draws the hierarchical observer.

        The observer shows the inverse projection of a node's cluster centers. It takes as input a list of projection
        groups. Each group consists of reconstructions of the individual cluster centers in one input space.
        The reconstruction of each cluster centers is an image with dimensions `HxWxC`.
        The array values can be float in [0,1] or uint8 in [0, 255].

        The properties are grouped by projection group so that the groups can have e.g. scale set independently.
        """
        groups_data = []
        for images, group_params in zip(image_groups, params):
            assert 'scale' in group_params, 'Params must have scale set'

            image_data = []
            for img in images:
                height, width, _ = img.size()

                b64encoded = encode_image(img)

                values_params = {
                    'width': width,
                    'height': height,
                    'data': self.sanitize_json_numbers([[0]])
                }

                image_data.append({
                    'src': 'data:image/png;base64,' + b64encoded,
                    'values': values_params,
                })

            groups_data.append({
                'images': image_data,
                'params': group_params
            })

        content = {
            'groups_stacking': groups_stacking.name.lower(),
            'items_per_row': items_per_row,
            'groups': groups_data,
            'properties': properties
        }

        self._send_command_window(win, 'hierarchical', content)

    def cluster_observer(self, cluster_observer_data: ClusterObserverData, properties: List[ObserverPropertiesItem],
                         win: str):
        """
        Creates/updates Cluster Observer window.
        No specific `opts` are currently supported.
        """

        content = to_nested_dict(cluster_observer_data)
        content['properties'] = properties

        self._send_command_window(win, 'cluster_observer', content)

    def image(self, tensor: torch.Tensor, win: str):
        """Draw image

        Args:
            tensor: tensor of dims [height, width, 3] (last dimensions is RGB channels)
            win: window name
        """
        content = {
            'src': 'data:image/png;base64,' + encode_image(tensor),
        }
        self._send_command_window(win, 'image', content)

    def matplot(self, plot: plt, win: str):
        """Draw matplot chart"""

        buffer = io.StringIO()
        plot.savefig(buffer, format='svg')
        buffer.seek(0)
        svg = buffer.read()
        buffer.close()

        return self.svg(svg=svg, win=win)

    def text(self, text: str, win=None):
        """Print text to a window"""
        self._send_command_window(win, 'text', text)

    def properties(self, data, win=None):
        """
        This function shows editable properties in a pane.
        Properties are expected to be a List of Dicts e.g.:
        ```
            properties = [
                {'type': 'text', 'name': 'Text input', 'value': 'initial'},
                {'type': 'number', 'name': 'Number input', 'value': '12'},
                {'type': 'button', 'name': 'Button', 'value': 'Start'},
                {'type': 'checkbox', 'name': 'Checkbox', 'value': True},
                {'type': 'select', 'name': 'Select', 'value': 1,
                 'values': ['Red', 'Green', 'Blue']},
            ]
        ```
        Supported types:
         - text: string
         - number: decimal number
         - button: button labeled with "value"
         - checkbox: boolean value rendered as a checkbox
         - select: multiple values select box
            - `value`: id of selected value (zero based)
            - `values`: list of possible values

        Property item items:
         - type: property type
         - name: displayed name of property (should be unique)
         - value: value of the property
         - state: 'enabled'|'disabled'|'readonly' (if not present, 'enabled' is default)

        Callback are called on property value update:
         - `event_type`: `"PropertyUpdate"`
         - `propertyId`: position in the `properties` list
         - `value`: new value
        """
        self._send_command_window(win, 'properties', data)

    def _send_command_window(self, win: str, type: str, data: Any):
        self._connector.send_command('window', {'id': win, 'type': type, 'content': data})

    def svg(self, svg: str, win=None):
        """Draw svg image """

        svg_match = re.search('<svg .+</svg>', svg, re.DOTALL)
        assert svg_match is not None, 'SVG parse error'
        return self.text(text=svg_match.group(0), win=win)

    def remove_all_windows(self):
        self._connector.send_command('remove_all_windows', None)

    def close(self, win: str):
        """Close window"""

        self._connector.send_command('close', {'id': win})

    def register_event_handler(self, win_id: str, handler: Callable[[EventData], None]):
        self._connector.register_event_handler(win_id, handler)

    def register_request_handler(self, win_id: str, handler: Callable[[RequestData], None]):
        self._connector.register_request_handler(win_id, handler)

    def clear_event_handlers(self, win_id: str):
        self._connector.clear_event_handlers(win_id)

    def clear_request_handlers(self, win_id: str):
        self._connector.clear_request_handlers(win_id)
