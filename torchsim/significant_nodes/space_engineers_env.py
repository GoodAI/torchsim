import math

from typing import Tuple, List

from torchsim.core.datasets.dataset_se_base import SeDatasetSize
from torchsim.core.graph.connection import Connector
from torchsim.core.nodes import DatasetSeObjectsParams, DatasetSeObjectsNode, ConstantNode, SwitchNode
from torchsim.gui.validators import validate_predicate
from torchsim.significant_nodes.environment_base import EnvironmentParamsBase, EnvironmentBase


class SeEnvironmentParams(EnvironmentParamsBase):
    env_size: Tuple[int, int, int] = (24, 24, 3)
    n_shapes: int = DatasetSeObjectsNode.label_size()
    shapes: List[int] = (2, 3, 7)

    def __init__(self, env_size: Tuple[int, int]=(24, 24), shapes: List[int]=(2, 3, 7)):
        self.env_size = env_size + (3,)
        self.shapes = shapes
        self.n_shapes = DatasetSeObjectsNode.label_size()


class SEEnvironment(EnvironmentBase):

    SHAPES_N = 20

    def __init__(self, params: SeEnvironmentParams, name: str="SEEnvironment"):
        super().__init__(params, name)

        env_params = DatasetSeObjectsParams()
        if params.env_size == (24, 24, 3):
            env_params.dataset_size = SeDatasetSize.SIZE_24
        else:
            raise ValueError(f"Param env_size {params.env_size} is not supported.")

        env_params.class_filter = params.shapes
        se_node = DatasetSeObjectsNode(env_params)

        self.add_node(se_node)
        self.se_node = se_node

        Connector.connect(se_node.outputs.image_output, self.outputs.data.input)

        switch_node = SwitchNode(2)
        self.add_node(switch_node)
        self.switch_node = switch_node

        nan_node = ConstantNode(params.n_shapes, math.nan)
        self.add_node(nan_node)

        Connector.connect(se_node.outputs.task_to_agent_label, switch_node.inputs[0])
        Connector.connect(nan_node.outputs.output, switch_node.inputs[1])
        Connector.connect(switch_node.outputs.output, self.outputs.label.input)

    def switch_learning(self, on):
        self.switch_node.change_input(0 if on else 1)

    def get_correct_label_memory_block(self):
        return self.se_node.outputs.task_to_agent_label_ground_truth

    @staticmethod
    def validate_params(params: SeEnvironmentParams):
        validate_predicate(lambda: len(params.env_size) == 3 and params.env_size[0] >= 1 and params.env_size[1] >= 1 and params.env_size[
            2] == 3)
        validate_predicate(lambda: params.n_shapes > 0)
        validate_predicate(lambda: len(params.shapes) == params.n_shapes)
