import logging
import math

from enum import Enum
from typing import NamedTuple, List, Dict

import torch
from torchsim.core import FLOAT_TYPE_CPU

from torchsim.core.graph.inverse_pass_packet import InversePassInputPacket
from torchsim.core.memory.tensor_creator import TensorSurrogate
from torchsim.core.graph.hierarchical_observable_node import HierarchicalObservableNode
from torchsim.core.utils.inverse_projection_utils import get_inverse_projections_for_all_clusters
from torchsim.gui.observables import Observable, ObserverPropertiesItem, ObserverCallbacks, ObserverPropertiesBuilder
from torchsim.gui.observers.tensor_observable import TensorViewProjection, update_scale_to_respect_minimum_size, \
    TensorViewProjectionUIParams
from torchsim.gui.server.ui_server_connector import RequestData
from torchsim.gui.ui_utils import parse_bool

logger = logging.getLogger(__name__)


class HierarchicalObservableGroupsStacking(Enum):
    VERTICAL = 1
    HORIZONTAL = 2


class HierarchicalObservableParams(NamedTuple):
    scale: int
    projection: TensorViewProjectionUIParams


class HierarchicalObservableData(NamedTuple):
    groups_stacking: HierarchicalObservableGroupsStacking
    items_per_row: int
    image_groups: List[List[torch.Tensor]]
    params_groups: List[HierarchicalObservableParams]


class HierarchicalGroupProperties:
    _parent: 'HierarchicalObserver'
    scale: int = 4
    scale_set_by_user: int = 1
    is_rgb: bool = False
    tensor_view_projection: TensorViewProjection

    # is_extended = True

    # minimum observer size in pixels, used for automatic rescaling of observers which are too small
    #    MINIMAL_SIZE = 10

    def __init__(self, group_id: int, parent: 'HierarchicalObserver'):
        self._parent = parent
        self.group_id = group_id
        self.tensor_view_projection = TensorViewProjection(is_buffer=False)

    def project_and_scale(self, tensor):
        tensor, projection_params = self.tensor_view_projection.transform_tensor(tensor, self.is_rgb)
        self.scale = update_scale_to_respect_minimum_size(tensor, self._parent.minimal_group_size,
                                                          self.scale_set_by_user)
        return tensor, projection_params

    def get_properties(self):
        def update_scale(value):
            self.scale_set_by_user = int(value)
            return value

        def update_is_rgb(value):
            self.is_rgb = parse_bool(value)
            return value

        # def update_header(value):
        #     self.is_extended = parse_bool(value)
        #     return value

        properties = [
                         ObserverPropertiesItem("Scale", 'number', self.scale, update_scale),
                         ObserverPropertiesItem("RGB", 'checkbox', self.is_rgb, update_is_rgb)
                     ] + self.tensor_view_projection.get_properties()

        header_name = f'Group {self.group_id}'
        for prop in properties:
            prop.name = f"{header_name}.{prop.name}"

        return [
            self._parent.prop_builder.collapsible_header(header_name, True),
            # ObserverPropertiesItem(header_name, 'collapsible_header', self.is_extended, update_header),
            *properties]


# Make this more general, now it only works with HierarchicalObservableNode implementations, and assumes that there is
# the flock_size dimension present.
class HierarchicalObserver(Observable):
    # minimum observer size in pixels, used for automatic rescaling of observers which are too small

    # used to hack persistence
    _groups_max_count: int = 10
    _default_properties: Dict[int, HierarchicalGroupProperties]

    _properties: Dict[int, HierarchicalGroupProperties]
    _grouped_projections: List[List[torch.Tensor]]
    prop_builder: ObserverPropertiesBuilder
    _items_per_row: int = 1
    _groups_stacking: HierarchicalObservableGroupsStacking = HierarchicalObservableGroupsStacking.HORIZONTAL
    minimal_group_size: int = 10

    def __init__(self, node: HierarchicalObservableNode, expert_no: int):
        super().__init__()
        self._node = node
        self._expert_no = expert_no

        self._properties = {}

        self._grouped_projections = None
        self.prop_builder = ObserverPropertiesBuilder()

        # TODO HACK - persisted values are loaded prior to the node unit initialization which determines the number
        # of groups
        # properties not initialized - create dummy properties just to fix persistence
        self._default_properties = {i: HierarchicalGroupProperties(i, self)
                                    for i in range(self._groups_max_count)}

    def get_data(self) -> HierarchicalObservableData:
        self._grouped_projections = grouped_projections = get_inverse_projections_for_all_clusters(self._node,
                                                                                                   self._expert_no)

        for i in range(len(grouped_projections)):
            if i not in self._properties:
                if i < len(self._default_properties):
                    # New group - load default properties (with loaded data from persistence storage)
                    self._properties[i] = self._default_properties[i]
                else:
                    logger.warning(f'Hierarchical observer {self._node.name_with_id}.expert_{self._expert_no}: Too '
                                   f'many groups found, values will not be persisted. Increase self._groups_max_count.')
                    self._properties[i] = HierarchicalGroupProperties(i, self)

        image_groups = []
        params_groups = []
        for i, projection_group in enumerate(grouped_projections):
            group_properties = self._properties[i]

            group_images = []
            group_projection_params = None
            for projection in projection_group:
                tensor, projection_params = group_properties.project_and_scale(projection)
                group_images.append(tensor)

                # These are not appended, they are all the same.
                group_projection_params = projection_params

            image_groups.append(group_images)
            params_groups.append(
                HierarchicalObservableParams(scale=group_properties.scale, projection=group_projection_params))

        return HierarchicalObservableData(self._groups_stacking, self._items_per_row, image_groups, params_groups)

    def get_properties(self) -> List[ObserverPropertiesItem]:
        def update_items_per_row(value: int):
            self._items_per_row = value

        def update_minimal_group_size(value: int):
            self.minimal_group_size = value

        def update_groups_stacking(value):
            self._groups_stacking = value

        properties = [
            self.prop_builder.collapsible_header('Global', True),
            self.prop_builder.select('Global.Groups stacking',
                                     self._groups_stacking,
                                     update_groups_stacking,
                                     HierarchicalObservableGroupsStacking),
            self.prop_builder.number_int('Global.Items per row', self._items_per_row,
                                         update_items_per_row),
            self.prop_builder.number_int('Global.Minimal size', self.minimal_group_size,
                                         update_minimal_group_size)
        ]

        if len(self._properties) == 0:
            # Hack for property persistence - this branch is visited when the observer system is initialized
            # and persisted values are loaded into the properties - the placeholder properties are needed
            for group_id in self._default_properties:
                properties.extend(self._default_properties[group_id].get_properties())
        else:
            for group_id in self._properties:
                properties.extend(self._properties[group_id].get_properties())

        return properties

    def request_callback(self, request_data: RequestData):
        data = request_data.data
        x = int(data['x'])
        y = int(data['y'])
        group_idx = int(data['group_idx'])
        image_idx = int(data['image_idx'])

        lookup_not_possible = (self._grouped_projections is None) or (
                len(self._grouped_projections) < group_idx + 1) or (
                                      group_idx not in self._properties) or self._properties[group_idx].is_rgb or (
                                      len(self._grouped_projections[group_idx]) < image_idx + 1)

        if lookup_not_possible:
            value = float('nan')
        else:
            value = self._properties[group_idx].tensor_view_projection.value_at(
                self._grouped_projections[group_idx][image_idx], x, y)

        return {
            "value": 'NaN' if math.isnan(value) else value
        }

    def get_callbacks(self) -> ObserverCallbacks:
        return ObserverCallbacks().add_request(self.request_callback)
