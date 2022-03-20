from collections import Counter
from random import Random
from typing import Tuple, TypeVar, Union, List, Dict, Collection

import numpy as np
from gym import spaces

from autograph.lib.envs.gridenv import GridEnv
from autograph.lib.envs.saveloadenv import SaveLoadEnv
from autograph.lib.util import element_add


class MineWorldTileType:
    """A single special tile in the mine world"""

    def __init__(self, consumable: bool, inventory_modifier: Counter, ap_name: str, grid_letter: str,
                 wall: bool = False):
        """
        :param consumable: Does this tile disappear after being activated
        :param inventory_modifier: How does this modify the inventory (e.g. wood -2, desk +1)
        :param ap_name: What atomic proposition should be true the round that this tile is activated
        :param grid_letter: What letter should be displayed on the grid
        """
        self.consumable = consumable
        self.inventory = inventory_modifier
        self.ap_name = ap_name
        self.grid_letter = grid_letter
        self.wall = wall

    def apply_inventory(self, prev_inventory: Counter):
        """
        Get the new inventory of the player after interacting with this tile, or errors if the player is unable to
        interact with the tile
        :param prev_inventory: The current inventory of the player
        """

        # Apply all the inventory changes and make sure that no item is negative
        new_inv = prev_inventory.copy()
        new_inv.update(self.inventory)
        if any([(new_inv[i] < 0) for i in new_inv]):
            raise ValueError()
        else:
            return new_inv

    @staticmethod
    def from_dict(dict):
        wall = dict.get("wall", False)
        return MineWorldTileType(consumable=dict["consumable"], inventory_modifier=Counter(dict["inventory_modifier"]),
                                 ap_name=dict["ap_name"], grid_letter=dict["grid_letter"], wall=wall)


T = TypeVar("T")
MaybeRand = Union[T, str]


class TilePlacement:
    def __init__(self, tile: MineWorldTileType, fixed_placements: Collection[Tuple[int, int]] = tuple(),
                 random_placements: int = 0):
        self.tile = tile
        self.fixed_placements = fixed_placements
        self.random_placements = random_placements

    @staticmethod
    def from_dict(dict):
        tile = MineWorldTileType.from_dict(dict["tile"])
        fixed_raw = dict.get("fixed_placements", [])
        fixed_placements = [tuple(coord) for coord in fixed_raw]
        random_placements = dict.get("random_placements", 0)
        return TilePlacement(tile=tile,
                             fixed_placements=fixed_placements,
                             random_placements=random_placements)


class InventoryItemConfig:
    def __init__(self, name: str, default_quantity: int, capacity: int):
        """
        :param name: Name of the item, like wood or iron
        :param default_quantity: How many of these items to start with
        :param capacity: Maximum amount of this item the agent can hold. Also used for scaling of NN inputs.
        """
        self.name = name
        self.default_quantity = default_quantity
        self.capacity = capacity

    @staticmethod
    def from_dict(dict):
        return InventoryItemConfig(**dict)


class MineWorldConfig:
    # init_positions list of tuples that contain the coordinate and the inventory configuration
    def __init__(self, shape: Tuple[int, int], num_teams: int, num_agents: int,
                 init_positions: List[Tuple[int, int]],
                 placements: List[TilePlacement],
                 inventories: List[List[InventoryItemConfig]],
                 player_num: int,
                 partial: bool,
                 partial_dist: int,
                 init_directions: List[int]):
        self.placements = placements
        self.shape = shape
        self.num_teams = num_teams
        self.num_agents = num_agents
        # teams split using modulo num_teams on list of positions
        self.init_positions = init_positions
        self.inventories = inventories
        self.player_num = player_num
        self.partial = partial
        self.partial_dist = partial_dist
        self.init_directions = init_directions

    @staticmethod
    def from_dict(dict, player_num: int):
        shape = tuple(dict["shape"])
        # grab new list from the json listpositions_aut_20_adv
        ips = list(dict["init_positions"])
        num_agents = dict["num_agents"]
        # may be necessary if we want 3 agents but only send initial positions for 2, 1, or 0
        num_teams = dict["num_teams"]
        placement = [TilePlacement.from_dict(i) for i in dict["placements"]]
        # send the inventories with the initial positions
        inventory_names = [k for k in dict["inventories"]]
        init_directions = list(dict["init_directions"])
        inventories = list()
        for iN in inventory_names:
            inventories.append(list(map(InventoryItemConfig.from_dict, dict["inventories"][iN])))

        partial = dict["partial_observability"]
        partial_distance = dict["partial_distance"]
        return MineWorldConfig(shape=shape, num_teams=num_teams, num_agents=num_agents,
                               init_positions=ips, placements=placement, inventories=inventories,
                               player_num=player_num, partial=partial, partial_dist=partial_distance,
                               init_directions=init_directions,
                               )


class MineWorldEnv(GridEnv, SaveLoadEnv):
    @staticmethod
    def from_dict(dict, player_num):
        x = 0
        return MineWorldEnv(MineWorldConfig.from_dict(dict, player_num))

    def __init__(self, config: MineWorldConfig, *args, **kwargs):
        super().__init__(shape=config.shape, *args, **kwargs)
        self.action_space = spaces.Discrete(5)
        # is this valid?
        # setting class parameter to one of its functions
        # self.action_space = self.get_action_space()
        # we also give each agent their action space
        # different actions for different agents, can we make this equal to a function
        # that gets the action space of agent at self.turnn()
        self.agents = []
        self.inventories = config.inventories
        # change this to deal with noninputted agents.
        # for now lets assume we have all the agents we want in json

        for i in range(0, config.num_agents):
            inv = config.inventories[i]
            init_posit = config.init_positions[i]
            # each agent has it's own inventory, initial position, index
            self.agents.append(MineWorldAgent(inv, init_posit, i, config.init_directions[i]))

        self.positions = config.init_positions
        self.config = config
        self.rand = Random()
        self.game_step = -1
        self.done = True
        self.hideAndSeek = True
        self.special_tiles: Dict[Tuple[int, int], MineWorldTileType] = dict()
        self.inventoryCounters: Dict[int, Counter] = dict()
        for i in range(0, config.num_agents):
            self.inventoryCounters[i] = self.agents[i].inventory
        self.turnn = 0

        self.partial = config.partial
        self.partial_dist = config.partial_dist
        self.player_num = config.player_num

    def get_agents(self):
        return self.agents

    # assume turn is valid turn from 0 to n-1 players
    def get_agent(self, turn):
        return self.agents[turn]

    # how could we use this with openAI gym
    def get_action_space(self):
        return self.agents[self.get_turnn()].action_space

    def get_turnn(self):
        return self.turnn

    def step(self, action: int):
        self.game_step += 1
        # this function only updates movement and inventory in the env
        stepper = self.agents[self.turnn]
        #if self.hideAndSeek and self.game_step < 200:
        #    if stepper.playerNum % 2 == 1:
        #       action = 4
        # should we change turn before or after this
        self.turnn = (self.get_turnn() + 1) % len(self.agents)
        assert stepper.action_space.contains(action)
        assert not self.done

        atomic_propositions = set()

        if action == 0:

            # Move forward
            action_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            new_place = element_add(stepper.position, action_offsets[stepper.direction])
            can_move = self._in_bounds(new_place)

            if new_place in self.special_tiles:
                if self.special_tiles[new_place].wall:
                    can_move = False

            if can_move:
                stepper.position = new_place

        elif action < 3:
            # for changing the facing direction of the agent
            add = 1 if action == 1 else -1
            dir = stepper.direction
            dir += add
            if dir < 0:
                dir += 4
            dir %= 4
            stepper.direction = dir

        elif action == 3:
            # Interact with tile in front
            new_place = stepper.position

            action_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            new_place = element_add(stepper.position, action_offsets[stepper.direction])
            can_move = self._in_bounds(new_place)
            if can_move:
                if new_place in self.special_tiles:
                    this_tile: MineWorldTileType = self.special_tiles[new_place]
                    if this_tile.ap_name == 'Box' and stepper.playerNum % 2 == 1:
                        # Tile in front is a box,
                        # seekers may break it given that at least two seekers are adjacent to it, and at least one of them holds a tool
                        for offset in action_offsets:
                            pos_check = element_add(new_place, offset)
                            count_seekers = 0
                            count_tools = 0
                            for x in range(len(self.agents)):
                                agent = self.agents[x]
                                if agent.position == pos_check:
                                    if x % 2 == 1:
                                        count_seekers += 1
                                        for inv_config in self.inventories[agent.playerNum]:
                                            if (inv_config.name == 'tool'):
                                                if (stepper.inventory[inv_config.name] > 0):
                                                    count_tools += 1
                            if count_seekers >= 2 and count_tools >= 1:
                                # can break it
                                del self.special_tiles[new_place]
                    else:
                        try:
                            new_inv = this_tile.apply_inventory(stepper.inventory)
                            for inv_config in self.inventories[stepper.playerNum]:
                                if new_inv[inv_config.name] > inv_config.capacity:
                                    new_inv[inv_config.name] = inv_config.capacity
                            stepper.inventory = new_inv
                            atomic_propositions.add(this_tile.ap_name)
                            self.inventoryCounters[stepper.playerNum] = stepper.inventory.copy()

                            if this_tile.consumable:
                                if (len(self.inventories) > 0):
                                    del self.special_tiles[new_place]
                        except ValueError:  # Couldn't apply inventory
                            pass
                else:
                    # Not on a special tile, that means tile in front is empty, hiders can attempt to place/build box here
                    if stepper.playerNum % 2 == 0:
                        built = False
                        for inv_config in self.inventories[stepper.playerNum]:
                            if inv_config.name == 'box':
                                # Place the box here if we have any boxes
                                if stepper.inventory[inv_config.name] > 0:
                                    built = True
                                    stepper.inventory[inv_config.name] -= 1
                                    self.inventoryCounters[stepper.playerNum] = stepper.inventory
                                    # set the tile to a box
                                    self.special_tiles[new_place] = self.config.placements[3].tile
                        if not built:
                            for inv_config in self.inventories[stepper.playerNum]:
                                if inv_config.name == 'tool':
                                    # Use the tool to build a box here if we have any
                                    if stepper.inventory[inv_config.name] > 0:
                                        built = True

                                        if self._in_bounds(new_place):
                                            #stepper.inventory[inv_config.name] -= 1
                                            self.inventoryCounters[stepper.playerNum] = stepper.inventory
                                            # set the tile to a box
                                            self.special_tiles[new_place] = self.config.placements[3].tile

        info = {
            'atomic_propositions': atomic_propositions,
            'inventory': stepper.inventory.copy()
        }

        if self.partial:
            return self._get_observation_partial(), 0, self.done, info
        else:
            # Reward is always 0 because it's minecraft. Exploration is reward in itself, and the only limit is imagination
            return self._get_observation(), 0, self.done, info

    def seed(self, seed=None):
        self.rand.seed(seed)

    def reset1(self):
        self.done = False
        self.special_tiles = self._get_tile_positioning()

        return self._get_observation()

    def reset(self):
        self.done = False
        self.turnn = 0  # first turn at 0?
        self.game_step = -1
        self.agents = []
        for i in range(0, self.config.num_agents):
            inv = self.config.inventories[i]
            init_posit = self.config.init_positions[i]
            # each agent has it's own inventory, initial position, index
            self.agents.append(MineWorldAgent(inv, init_posit, i, self.config.init_directions[i]))
            # self.inventories = self.config.inventories
        # change this to deal with noninputted agents.
        # for now lets assume we have all the agents we want in json
        self.special_tiles = self._get_tile_positioning()
        self.inventoryCounters: Dict[int, Counter] = dict()
        for i in range(0, len(self.agents)):
            self.inventoryCounters[i] = self.agents[i].inventory

        if self.partial:
            return self._get_observation_partial()
        else:
            return self._get_observation()

    def _get_tile_positioning(self) -> Dict[Tuple[int, int], MineWorldTileType]:

        tiles = {}

        for tile_type in self.config.placements:
            for fixed in tile_type.fixed_placements:
                tiles[fixed] = tile_type.tile

        all_spaces = set(np.ndindex(self.config.shape))
        open_spaces = all_spaces.difference(tiles.keys())
        if (0, 0) in open_spaces:
            open_spaces.remove((0, 0))

        for tile_type in self.config.placements:
            tile, num_placements = tile_type.tile, tile_type.random_placements
            spaces = self.rand.sample(open_spaces, num_placements)
            open_spaces.difference_update(spaces)

            for space in spaces:
                tiles[space] = tile

        return tiles

    def isViewable(self, distance):
        playerPosition = self.positions[self.player_num]
        playerDirection = self.agents[self.player_num].direction

        directionOffsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        viewableSpaces = set()

        #the perfect unobscured cone of vision to check for boxes to get partial observation
        for i in range(0, distance+1):
            offset = tuple(i*x for x in directionOffsets[playerDirection])
            straight_space = tuple(off+pp for pp, off in zip(playerPosition, offset))
            if self._in_bounds(straight_space):
                viewableSpaces.add(straight_space)

            for j in range(-i, i+1):
                if j == 0:
                    continue
                width_direction = directionOffsets[(playerDirection + 1) % 4]
                width_space = tuple(ssp+(j*wd)
                                    for ssp, wd in zip(straight_space, width_direction))
                if self._in_bounds(width_space):
                    viewableSpaces.add(width_space)

        return viewableSpaces

        # modified for multiagent

    def _get_observation_partial(self):
        # env returning the state to the MCTS
        self.positions = tuple(agent.position for agent in self.agents)
        cone_positions = list(agent.position for agent in self.agents)

        playerPosition = self.positions[self.player_num]
        playerDirection = self.agents[self.player_num].direction

        viewableSet = self.isViewable(self.partial_dist)

        for idx in range(0, len(cone_positions)):
            if cone_positions[idx] not in viewableSet:
                cone_positions[idx] = (-10, -10)

        # not yet a frozenset, so we can remove in the removal part of the function
        tiles = tuple(
            set(space for space, content in self.special_tiles.items()
                if content is placement.tile and space in viewableSet) for
            placement in self.config.placements)

        directionOffsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        mustBeRemoved = False
        obscured_dist = self.partial_dist

        removeByLevel = list()

        # carry information along the "direction of vision" from agent outwards
        # we must know if object or player on level prior obscures view of objects on current level
        # On the sides of the cone, an object will block out all objects behind and to the far side of the cone.
        removeByLevel.append([0, 0])
        for level in range(1, self.partial_dist + 2):
            removeByLevel.append([0, 0])


        for i in range(1, self.partial_dist + 1):
            # removeByLevel[i][0] = removeByLevel[i-1][0]
            # removeByLevel[i][1] = removeByLevel[i-1][1]
            offset = tuple(i * x for x in directionOffsets[playerDirection])
            newSpace = tuple(off + pp for pp, off in zip(playerPosition, offset))
            must_remove_temp = mustBeRemoved
            # deal with line directly in front
            for tileType in tiles:

                if not must_remove_temp and newSpace in tileType:
                    mustBeRemoved = True
                    obscured_dist = i


                elif must_remove_temp and newSpace in tileType:
                    viewableSet.remove(newSpace)
                    tileType.remove(newSpace)
                    continue

                for idx in range(0, len(cone_positions)):
                    if must_remove_temp and cone_positions[idx] == newSpace:
                        cone_positions[idx] = (-10, -10)
                    if not must_remove_temp and cone_positions[idx] == newSpace:
                        mustBeRemoved = True
                        obscured_dist = i

            #deal with one width direction
            for j in range(-i, 0):
                width_direction = directionOffsets[(playerDirection + 1) % 4]
                width_space = tuple(newsp + (j * wd)
                                    for newsp, wd in zip(newSpace, width_direction))
                for tileType in tiles:
                    if width_space in tileType:
                        if removeByLevel[i][0] <= j:
                            tileType.remove(width_space)
                            viewableSet.remove(width_space)
                        else:
                            for k in range(i + 1, len(removeByLevel)):
                                removeByLevel[k][0] = min(j - 1, removeByLevel[i][0])
                for idx in range(0, len(self.positions)):
                    if cone_positions[idx] == width_space:
                        if removeByLevel[i][0] > j:
                            # block out the stuff behind it and to the width direction of it
                            for k in range(i + 1, len(removeByLevel)):
                                removeByLevel[k][0] = min(j - 1, removeByLevel[i][0])
                        elif removeByLevel[i][0] <= j:
                            cone_positions[idx] = (-10, -10)


            #the other width direction
            for j in range(1, i + 1):
                width_direction = directionOffsets[(playerDirection + 1) % 4]
                width_space = tuple(newsp + (j * wd)
                                    for newsp, wd in zip(newSpace, width_direction))
                for tileType in tiles:
                    if width_space in tileType:
                        if removeByLevel[i][1] >= j:
                            tileType.remove(width_space)
                            viewableSet.remove(width_space)
                        else:
                            for k in range(i + 1, len(removeByLevel)):
                                removeByLevel[k][1] = max(j + 1, removeByLevel[i][1])
                for idx in range(0, len(cone_positions)):
                    if cone_positions[idx] == width_space:
                        if removeByLevel[i][1] <= j:
                            # block out the stuff behind it and to the width direction of it
                            for k in range(i + 1, len(removeByLevel)):
                                removeByLevel[k][1] = max(j + 1, removeByLevel[i][1])
                        elif removeByLevel[i][1] > j:
                            cone_positions[idx] = (-10, -10)

        viewableTiles = list()

        for tileType in tiles:
            viewableTiles.append(frozenset(tileType))

        mult_inv_ratios = list()

        for i in range(0, len(self.agents)):
            inventory = self.agents[i].inventory
            if cone_positions[i] != (-10, -10):
                inv_ratios = tuple(
                    inventory[inv_config.name] / inv_config.capacity for inv_config in self.config.inventories[i])
            else:
                # send a -10 if we can't see their inventory
                inv_ratios = tuple(
                    -10 for inv_config in self.config.inventories[i])
            mult_inv_ratios.append(inv_ratios)

            # cant see them, cant see their inventory

        # agent1 moves into a state where it is agent2 turn.
        # which turn do we send from observation

        return (
            # tuple(self.agents),
            tuple(cone_positions),
            tuple(viewableTiles),
            tuple(mult_inv_ratios),
            self.turnn,
        )

    def get_obs_full(self):
        return self._get_observation()

    # modified for multiagent
    def _get_observation(self):
        # env returning the state to the MCTS
        tiles = tuple(
            frozenset(space for space, content in self.special_tiles.items() if content is placement.tile) for
            placement in self.config.placements)

        mult_inv_ratios = list()
        self.positions = tuple(agent.position for agent in self.agents)

        for i in range(0, len(self.agents)):
            inventory = self.agents[i].inventory
            inv_ratios = tuple(
                inventory[inv_config.name] / inv_config.capacity for inv_config in self.config.inventories[i])
            mult_inv_ratios.append(inv_ratios)

        # agent1 moves into a state where it is agent2 turn.
        # which turn do we send from observation

        return (
            # tuple(self.agents),
            self.positions,
            tiles,
            tuple(mult_inv_ratios),
            self.turnn,
        )

    # modified for multiagent - brett
    def render(self, mode='human'):
        def render_func(x, y):
            agentsHere = []
            for i in range(0, len(self.agents)):
                if self.positions[i] == (x, y):
                    agentsHere.append(self.agents[i])
            if len(agentsHere) > 1:
                # prints number of agents at this point
                agent_str = "x" + str(len(agentsHere))
            elif len(agentsHere) == 1:
                agent_str = str(agentsHere[0].playerNum)
            else:
                agent_str = " "

            # agent_str   = "1" if self.position   == (x, y) else " "
            tile_str = self.special_tiles[(x, y)].grid_letter if (x, y) in self.special_tiles else " "
            directionStrings = ["'","-", ","]

            if len(agentsHere) == 1:
                if agentsHere[0].direction <= 2:
                    return agent_str + directionStrings[agentsHere[0].direction] + tile_str, False, False
                else:
                    return directionStrings[1] + agent_str + tile_str, False, False



            return agent_str + tile_str, False, False
        print(self._render(render_func, 2), end="")
        # print each inventory
        for agent in self.agents:
            print(agent.playerNum, "at ", agent.position, " has", dict(self.inventoryCounters[agent.playerNum]))

    def save_state(self):
        # return self.agents.copy(), self.positions, self.turnn, self.done, self.special_tiles.copy() #  , self.inventory.copy()
        return self.positions, self.inventoryCounters.copy(), self.turnn, self.done, self.special_tiles.copy()  # , self.inventory.copy()

    def load_state(self, state):
        # self.agents, self.positions, self.turnn, self.done, spec_tile = state
        self.positions, inventoryCounters, self.turnn, self.done, spec_tile = state
        for i in range(0, len(self.agents)):
            self.agents[i].position = self.positions[i]
            self.agents[i].inventory = inventoryCounters[i].copy()
        self.special_tiles = spec_tile.copy()
        self.inventoryCounters: Dict[int, Counter] = dict()
        for i in range(0, len(self.agents)):
            self.inventoryCounters[i] = self.agents[i].inventory.copy()
        # self.inventory = inv.copy()


class MineWorldAgent:
    """single agents in the mine world"""

    def __init__(self, inventory: List[InventoryItemConfig], initial_position: Tuple[int, int], playerNum: int,
                 direction: int, *args,
                 **kwargs):
        self.action_space = spaces.Discrete(5)
        self.default_inventory = Counter(
            {inv_type.name: inv_type.default_quantity for inv_type in inventory})
        """
        0 Move forward,
        1 Turn clockwise,
        2 Turn counterclockwise,
        3 Interact with tile in front
          """

        self.position = tuple(initial_position)
        # self.inventory = Counter()
        self.inventory = self.default_inventory
        self.playerNum = playerNum

        # can only move forward in facing direction (should be 0, 1, 2, 3) (U R D L)
        self.direction = direction

    # TODO: proper hash function?
    def __hash__(self):
        return hash(self.position) + hash(tuple(self.inventory.items()))

    def __eq__(self, obj: T):
        return (self.position == obj.position and
                self.playerNum == obj.playerNum and
                self.inventory == obj.inventory
                )

    def changePosition(self, pos: Tuple[int, int]):
        self.position = pos