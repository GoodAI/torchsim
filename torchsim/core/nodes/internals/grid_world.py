import logging
from dataclasses import dataclass
import random
from enum import Enum
from typing import List

import numpy as np
import torch
from ruamel import yaml
from torch.nn.functional import interpolate

from torchsim.core.actions import AgentActionsDescriptor
from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.unit import Unit
from torchsim.core.memory.tensor_creator import TensorCreator, MeasuringCreator, TensorSurrogate
from torchsim.core.models.expert_params import ParamsBase
from torchsim.gui.validators import validate_predicate

logger = logging.getLogger(__name__)

class ResetStrategy(Enum):
    AGENT_SPAWN = 0
    ANYWHERE = 1
    SELECTED = 2


class GridWorldActionDescriptor(AgentActionsDescriptor):
    _action_names = [
        'UP',
        'DOWN',
        'RIGHT',
        'LEFT',
    ]

    ACTION_COUNT = len(_action_names)

    def parse_actions(self, actions: np.array) -> List[bool]:
        return [val > 0 for val in actions]

    def action_names(self) -> List[str]:
        return self._action_names

    @staticmethod
    def contains_action(actions: torch.Tensor, action: str) -> bool:
        if action == "UP" and actions[0] == 1:
            return True
        elif action == "DOWN" and actions[1] == 1:
            return True
        elif action == "RIGHT" and actions[2] == 1:
            return True
        elif action == "LEFT" and actions[3] == 1:
            return True
        else:
            return False


@dataclass
class GridWorldParams(ParamsBase):
    """Class used for configuring the GridWorld based on script 'grid_world_maps.yaml'.

    Args:
        reset_strategy: Dictates how the agent position is reset when it obtains reward, or hits a teleporter ('t').
        reward_switching: Indicates if the reward should be moved around. In this case, it is cycled (with p=.5) around a number of
                          teleporter locations and a clue (indicated on the map as 'c' or 'C') is changed to the opposite
                          case.

        Reward_switiching is currently supported for only two rewards, as there are only two clues which we can render at
        present.

    """
    tile_size: int = 9
    PATH_TO_MAP_SCRIPT: str = './torchsim/core/scripts/grid_world_maps.yaml'
    map_name: str = 'MapA'
    world_map = None
    world_width: int = None
    world_height: int = None
    egocentric_width: int = 3
    egocentric_height: int = 3
    agent_pos: List[int] = None
    starting_agent_pos = None
    reward: float = 100.0
    reset_strategy: ResetStrategy = None
    anywhere_valid_positions: List[List[int]] = None
    selected_valid_positions: List[List[int]] = None
    selected_teleportation_positions: List[List[int]] = None
    clue_position: List[List[int]] = None
    reward_switching: bool = False

    def __init__(self,
                 map_name: str = 'MapE',
                 reset_strategy: ResetStrategy = ResetStrategy.AGENT_SPAWN,
                 reward_switching: bool = False):

        self.map_name = map_name
        self.reset_strategy = reset_strategy
        self.reward_switching = reward_switching
        if self.reward_switching:
            self.reset_strategy = ResetStrategy.AGENT_SPAWN

        try:
            # read the script file
            map_data = self.load_map_script().get(self.map_name)

            # get the map matrix, its dimensions and agent position
            # NOTE: The map is read row major, so the correct coord system is (y, x)
            self.world_map = map_data['Map']
            self.world_width = list(map_data['Width'])[0]
            self.world_height = list(map_data['Height'])[0]
            self.agent_pos = self.find_agent()

        except IOError:
            IOError('File does not exist or has unrecognizable format!')

    def load_map_script(self):
        with open(self.PATH_TO_MAP_SCRIPT, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
            return data_loaded

    def find_agent(self):
        # search for 'A' in the world map
        agents_list = [[x, y] for x, l in enumerate(self.world_map) for y, v in enumerate(l) if v == 'A']
        if len(agents_list) == 0:
            raise Exception("No agent position found on the map!")
        if len(agents_list) > 1:
            raise Exception("More than one agent positions found on the map!")

        self.anywhere_valid_positions = [[y, x] for x, l in enumerate(self.world_map) for y, v in enumerate(l) if
                                v in [0, 'A', 's']]
        self.selected_valid_positions = [[y, x] for x, l in enumerate(self.world_map) for y, v in enumerate(l) if
                                v in ['s', 'A']]

        self.selected_teleportation_positions = [[y, x] for x, l in enumerate(self.world_map) for y, v in enumerate(l) if
                                v in ['R', 't']]

        self.clue_position = [[y, x] for x, l in enumerate(self.world_map) for y, v in enumerate(l) if
                                v in ['C', 'c']]

        # remove the agent's marker from the map
        x = agents_list[0][0]
        y = agents_list[0][1]
        self.world_map[x][y] = 0
        # Coords are flipped because rows are accessed first.
        self.starting_agent_pos = (y, x)

        return agents_list[0]

    def _get_ego(self, x, y):
        surrounding = [self.world_map[x_][y_] for x_ in range(x-1, x+2) for y_ in range(y-1, y+2)]
        for i in range(len(surrounding)):
            surrounding[i] = surrounding[i] if surrounding[i] not in ('R', 'A', 's') else 0
        return tuple(surrounding)

    def get_n_unique_visible_egocentric_states(self) -> int:
        ego_renders = []
        for x, l in enumerate(self.world_map):
            for y, v in enumerate(l):
                if v in [0, 'A', 's']:
                    ego_renders.append(self._get_ego(x, y))

        unique = set(ego_renders)
        return int(len(unique))


class GridWorld(Unit):
    """Creates the world, then presents each image and position of the agent in the world."""

    bit_map: torch.Tensor
    params: GridWorldParams

    circle_wall_template: List = [[0., 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 1, 1, 0, 0, 0],
                                  [0, 0, 1, 0, 1, 0, 1, 0, 0],
                                  [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                  [0, 1, 1, 1, 1, 1, 1, 1, 0],
                                  [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                  [0, 0, 1, 0, 1, 0, 1, 0, 0],
                                  [0, 0, 0, 1, 1, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    triangle_wall_template: List = [[0., 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
                                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 1, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    circle_template: List = [[0., 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 1, 1, 0, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 0, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 0, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 0, 1, 1, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    def __init__(self, creator: TensorCreator, params: GridWorldParams):
        super().__init__(creator.device)

        self.validate_params(params)

        self.params = params

        # render a world map
        tile_size = self.params.tile_size
        width = self.params.world_width * self.params.tile_size
        height = self.params.world_height * self.params.tile_size
        self.bit_map = creator.zeros([height, width], dtype=torch.uint8)

        if isinstance(creator, MeasuringCreator):
            self.last_image = creator.zeros([height, width], dtype=self._float_dtype, device=self._device)
            self.circle = creator.zeros([tile_size, tile_size], dtype=self._float_dtype, device=self._device)
            self.circle_wall = creator.zeros([tile_size, tile_size, 3], dtype=self._float_dtype, device=self._device)
            self.triangle_wall = creator.zeros([tile_size, tile_size, 3], dtype=self._float_dtype, device=self._device)
        else:

            self.circle = interpolate(creator.tensor(self.circle_template).view(1, 1, 9, 9),
                                      size=[tile_size, tile_size]).view(tile_size, tile_size).type(torch.uint8)
            self.triangle_wall = interpolate(creator.tensor(self.triangle_wall_template).view(1, 1, 9, 9),
                                      size=[tile_size, tile_size]).view(tile_size, tile_size).type(torch.uint8)
            self.circle_wall = interpolate(creator.tensor(self.circle_wall_template).view(1, 1, 9, 9),
                                      size=[tile_size, tile_size]).view(tile_size, tile_size).type(torch.uint8)

        self.bit_map_last_action = creator.zeros([height + 1, width], dtype=self._float_dtype, device=self._device)
        self.bit_map_padding = width - 4

        self.render_map(self.bit_map)

        self.pos = creator.tensor(self.params.agent_pos[::-1], dtype=self._float_dtype, device=self._device)
        self.last_position = creator.zeros(2, dtype=self._float_dtype, device=self._device)
        self.last_position_one_hot_matrix = creator.zeros((params.world_height, params.world_width),
                                                          dtype=self._float_dtype, device=self._device)
        self.last_action = creator.zeros(4, dtype=self._float_dtype, device=self._device)
        self.reward = creator.zeros(1, dtype=self._float_dtype, device=self._device)
        width = self.params.egocentric_width * self.params.tile_size
        height = self.params.egocentric_height * self.params.tile_size

        self.egocentric_image = creator.zeros((width, height), dtype=self._float_dtype, device=self._device)
        self.ego_last_action = creator.zeros((width+1, height), dtype=self._float_dtype, device=self._device)
        self.ego_padding = height - 4

        if not isinstance(creator, MeasuringCreator):
            # Initial rendering of the outputs, all further renderings will be just updates
            self.last_image = creator.tensor(self.bit_map, dtype=self._float_dtype, device=self._device)
            self.last_position.copy_(self.pos)
            self._render_outputs()

    @staticmethod
    def validate_params(params: GridWorldParams):
        validate_predicate(lambda: params.tile_size >= 1)
        validate_predicate(lambda: params.egocentric_width >= 1)
        validate_predicate(lambda: params.egocentric_height >= 1)

    def step(self, action: torch.Tensor):
        # Unscaled top-left corner position for renderer which scales it and adds a tile_size to it (therefore -1)
        width = self.params.world_width - 1
        height = self.params.world_height - 1

        # try perform movement within the world boundaries
        if GridWorldActionDescriptor.contains_action(action, "UP") and self.pos[1] > 0:
            self.pos[1] -= 1
        if GridWorldActionDescriptor.contains_action(action, "DOWN") and self.pos[1] < height:
            self.pos[1] += 1
        if GridWorldActionDescriptor.contains_action(action, "RIGHT") and self.pos[0] < width:
            self.pos[0] += 1
        if GridWorldActionDescriptor.contains_action(action, "LEFT") and self.pos[0] > 0:
            self.pos[0] -= 1

        # detect collisions with map objects
        x = int(self.pos[0])
        y = int(self.pos[1])
        if self.params.world_map[y][x] not in [0, 'R', 's', 't']:
            self.pos.copy_(self.last_position)
            self.reward.fill_(0)
        else:
            # If we can get into this location...
            if (self.last_position != self.pos).any():
                # If we hit an 'R', then get some reward and reset the agent
                if self.params.world_map[y][x] == 'R':
                    self.reward.fill_(self.params.reward)
                    self._reset_agent_pos()
                    if self.params.reward_switching:
                        val = random.randint(0, 10)
                        if val < 5:
                            self._switch_rewards_and_clues()

                elif self.params.world_map[y][x] == 't':
                    self.reward.fill_(0)
                    self._reset_agent_pos()
                    if self.params.reward_switching:
                        val = random.randint(0, 10)
                        if val < 5:
                            self._switch_rewards_and_clues()

                else:
                    self.reward.fill_(0)

                self._render_outputs()

        self.last_action.copy_(action)

        ego_action = torch.cat([self.last_action, torch.zeros(self.ego_padding, dtype=self._float_dtype, device=self._device)], dim=0).view(1, -1)
        bitmap_action = torch.cat([self.last_action, torch.zeros(self.bit_map_padding, dtype=self._float_dtype, device=self._device)], dim=0).view(1, -1)

        self.ego_last_action.copy_(torch.cat([self.egocentric_image, ego_action], dim=0))
        self.bit_map_last_action.copy_(torch.cat([self.last_image, bitmap_action], dim=0))

        return action

    def _switch_rewards_and_clues(self):
        clue_pos = self.params.clue_position[0]

        first_loc = self.params.selected_teleportation_positions[0]
        second_loc = self.params.selected_teleportation_positions[1]

        if self.params.world_map[first_loc[1]][first_loc[0]] == 'R':
            self.params.world_map[first_loc[1]][first_loc[0]] = 't'
            self.params.world_map[second_loc[1]][second_loc[0]] = 'R'
            self.params.world_map[clue_pos[1]][clue_pos[0]] = 'c'

        else:
            self.params.world_map[first_loc[1]][first_loc[0]] = 'R'
            self.params.world_map[second_loc[1]][second_loc[0]] = 't'
            self.params.world_map[clue_pos[1]][clue_pos[0]] = 'C'

        self.render_map(self.bit_map)
        self.last_image.copy_(self.bit_map)

    def _reset_agent_pos(self):
        if self.params.reset_strategy is ResetStrategy.AGENT_SPAWN:
            self.pos[0] = self.params.starting_agent_pos[0]
            self.pos[1] = self.params.starting_agent_pos[1]
        elif self.params.reset_strategy is ResetStrategy.SELECTED:
            pos = random.choice(self.params.selected_valid_positions)
            self.pos[0] = pos[0]
            self.pos[1] = pos[1]
        else:
            pos = random.choice(self.params.anywhere_valid_positions)
            self.pos[0] = pos[0]
            self.pos[1] = pos[1]

    def _render_outputs(self):
        img = self.update_map(self.last_image)
        self.last_image.copy_(img)

        self.last_position.copy_(self.pos)

        self.last_position_one_hot_matrix.fill_(0)
        self.last_position_one_hot_matrix[int(self.pos[1]), int(self.pos[0])] = 1

        self._render_egocentric_view()

    def render_map(self, bitmap: torch.Tensor):
        for x in range(self.params.world_height):
            for y in range(self.params.world_width):
                if self.params.world_map[x][y] == 1:
                    self._render_empty_square((x * self.params.tile_size, y * self.params.tile_size), bitmap)
                elif self.params.world_map[x][y] == 'c':
                    self._render_empty_circle((x * self.params.tile_size, y * self.params.tile_size), bitmap)
                elif self.params.world_map[x][y] == 'C':
                    self._render_empty_triangle((x * self.params.tile_size, y * self.params.tile_size), bitmap)

    def update_map(self, bitmap: torch.Tensor):
        size = self.params.tile_size
        next_pos = self.pos.type(torch.int32) * size
        last_pos = self.last_position.type(torch.int32) * size

        self._render_blank_square(last_pos, bitmap)
        self._render_ball(next_pos, bitmap)

        return bitmap

    def _render_egocentric_view(self):
        shift_y = self.params.egocentric_height // 2
        shift_x = self.params.egocentric_width // 2
        tile_size = self.params.tile_size
        pos_y = int(self.pos[1].item())
        pos_x = int(self.pos[0].item())

        min_y = pos_y - shift_y
        max_y = pos_y + shift_y + 1
        min_x = pos_x - shift_x
        max_x = pos_x + shift_x + 1

        if min_x < 0 or max_x > self.params.world_width or min_y < 0 or max_y > self.params.world_height:
            raise NotImplementedError("Egocentric view currently does not work if the agent would view over the edge"
                                      " of the map.")

        self.egocentric_image.copy_(self.last_image[(min_y * tile_size):(max_y * tile_size),
                                    (min_x * tile_size):(max_x * tile_size)])

    def _render_square(self, pos: np.array, bitmap: torch.Tensor):
        bitmap[pos[0]: self.params.tile_size,
        pos[1]: pos[0] + self.params.tile_size] = 1.0

    def _render_blank_square(self, pos: np.array, bitmap: torch.Tensor):
        bitmap[pos[1]: pos[1] + self.params.tile_size,
        pos[0]: pos[0] + self.params.tile_size] = 0.0

    def _render_empty_square(self, pos: np.array, bitmap: torch.Tensor):
        bitmap[pos[0]: pos[0] + self.params.tile_size,
        pos[1]] = 1.0
        bitmap[pos[0]: pos[0] + self.params.tile_size,
        pos[1] + self.params.tile_size - 1] = 1.0

        bitmap[pos[0],
        pos[1]: pos[1] + self.params.tile_size] = 1.0
        bitmap[pos[0] + self.params.tile_size - 1,
        pos[1]: pos[1] + self.params.tile_size] = 1.0

    def _render_ball(self, pos: np.array, bitmap: torch.Tensor):
        bitmap[pos[1]: pos[1] + self.params.tile_size,
        pos[0]: pos[0] + self.params.tile_size] = self.circle

    def _render_empty_circle(self, pos: np.array, bitmap: torch.Tensor):
        if not isinstance(bitmap, TensorSurrogate):
            if bitmap[pos[0]: pos[0] + self.params.tile_size, pos[1]: pos[1] + self.params.tile_size].numel() > 0:
                bitmap[pos[0]: pos[0] + self.params.tile_size, pos[1]: pos[1] + self.params.tile_size] = self.circle_wall

    def _render_empty_triangle(self, pos: np.array, bitmap: torch.Tensor):
        if not isinstance(bitmap, TensorSurrogate):
            if bitmap[pos[0]: pos[0] + self.params.tile_size, pos[1]: pos[1] + self.params.tile_size].numel() > 0:
                bitmap[pos[0]: pos[0] + self.params.tile_size, pos[1]: pos[1] + self.params.tile_size] = self.triangle_wall


class GridWorldOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output_image = self.create("Output_image")
        self.egocentric_image = self.create("Egocentric_image")
        self.output_pos = self.create("Output_pos")
        self.output_action = self.create("Output_action")
        self.output_pos_one_hot_matrix = self.create("Output_pos_one_hot_matrix")
        self.reward = self.create("Reward")
        self.output_image_action = self.create("Output_image_action")
        self.egocentric_image_action = self.create("Egocentric_image_action")

    def prepare_slots(self, unit: GridWorld):
        self.output_image.tensor = unit.last_image
        self.egocentric_image.tensor = unit.egocentric_image
        self.output_pos.tensor = unit.last_position
        self.output_action.tensor = unit.last_action
        self.output_pos_one_hot_matrix.tensor = unit.last_position_one_hot_matrix
        self.reward.tensor = unit.reward
        self.output_image_action.tensor = unit.bit_map_last_action
        self.egocentric_image_action.tensor = unit.ego_last_action


class GridWorldInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.agent_action = self.create("Action")
