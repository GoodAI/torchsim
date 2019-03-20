from typing import List

import logging

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import NodeGroupBase, GroupOutputs, GroupInputs
from torchsim.core.nodes.focus_node import FocusNode
from torchsim.core.nodes.motion_detection_node import MotionDetectionNode
from torchsim.core.nodes.salient_region_node import SalientRegionNode
from torchsim.gui.observables import ObserverPropertiesItem, disable_on_runtime
from torchsim.gui.validators import validate_positive_int

logger = logging.getLogger(__name__)


class BottomUpAttentionGroupInputs(GroupInputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.image = self.create("Image")


class BottomUpAttentionGroupOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.fof = self.create("FOF")
        self.focus_mask = self.create("Focus mask")
        self.coordinates = self.create("Coordinates")


class BottomUpAttentionGroup(NodeGroupBase[BottomUpAttentionGroupInputs, BottomUpAttentionGroupOutputs]):
    """Implements one-step bottom-up attention based on the simple motion detection
    """

    _is_fixed_region_used: bool
    _fixed_region_size: int

    def __init__(self):
        super().__init__("BottomUpAttentionGroup",
                         inputs=BottomUpAttentionGroupInputs(self),
                         outputs=BottomUpAttentionGroupOutputs(self))

        # produce motion map based on the motion detection
        self._motion_detection = MotionDetectionNode()
        self.add_node(self._motion_detection)

        # get motion map and produce focus coordinates
        self._salient_region = SalientRegionNode()
        self.add_node(self._salient_region)

        # get the coordinates and make focus
        self._focus_node = FocusNode()
        self.add_node(self._focus_node)

        # input -> detect motion
        Connector.connect(self.inputs.image.output,
                          self._motion_detection.inputs.input_image)

        # motion map -> salient region node (outputs (Y,X,height,width))
        Connector.connect(self._motion_detection.outputs.motion_map,
                          self._salient_region.inputs.input)

        # salient region (Y,X,height,width) -> focus node
        Connector.connect(self._salient_region.outputs.coordinates,
                          self._focus_node.inputs.coordinates)
        # image -> focus node
        Connector.connect(self.inputs.image.output,
                          self._focus_node.inputs.input_image)

        # both outputs of focus -> group outputs, focused coordinates -> output
        Connector.connect(self._focus_node.outputs.focus_output,
                          self.outputs.fof.input)
        Connector.connect(self._focus_node.outputs.focus_mask,
                          self.outputs.focus_mask.input)
        Connector.connect(self._salient_region.outputs.coordinates,
                          self.outputs.coordinates.input)

        # set the properties to all nodes identically
        self.use_fixed_region = False
        self.fixed_region_size = 12

    @property
    def fixed_region_size(self) -> int:
        return self._fixed_region_size

    @fixed_region_size.setter
    def fixed_region_size(self, value: int):
        validate_positive_int(value)
        self._fixed_region_size = value
        self._salient_region.fixed_region_size = value
        self._focus_node.trim_output_size = value

    @property
    def use_fixed_region(self) -> bool:
        return self._is_fixed_region_used

    @use_fixed_region.setter
    def use_fixed_region(self, value: bool):
        self._is_fixed_region_used = value
        self._salient_region.use_fixed_region_size = value
        self._focus_node.trim_output = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
                   self._prop_builder.auto('Use fixed region',
                                           type(self).use_fixed_region,
                                           edit_strategy=disable_on_runtime),
                   self._prop_builder.auto('Fixed region size',
                                           type(self).fixed_region_size,
                                           edit_strategy=disable_on_runtime)
               ] + super().get_properties()
