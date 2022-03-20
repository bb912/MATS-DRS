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
    def __init__(self, shape: Tuple[int, int], initial_position: Union[Tuple[int, int], None],initial_position_1: Union[Tuple[int, int], None],
                 placements: List[TilePlacement], inventory: List[InventoryItemConfig]):
        self.placements = placements
        self.shape = shape
        self.initial_position = initial_position
        self.initial_position_1 = initial_position_1
        self.inventory = inventory

    @staticmethod
    def from_dict(dict):
        shape = tuple(dict["shape"])
        ip = dict["initial_position"]
        initial_position = ip if ip is None else tuple(ip)
        ip_1 = dict["initial_position_1"]
        initial_position_1 = ip_1 if ip_1 is None else tuple(ip_1)
        placement = [TilePlacement.from_dict(i) for i in dict["placements"]]
        inventory = list(map(InventoryItemConfig.from_dict, dict["inventory"]))

        return MineWorldConfig(shape=shape, initial_position=initial_position, initial_position_1=initial_position_1,placements=placement,
                               inventory=inventory)


class MineWorldEnv(GridEnv, SaveLoadEnv):

    @staticmethod
    def from_dict(dict):
        return MineWorldEnv(MineWorldConfig.from_dict(dict))

    def __init__(self, config: MineWorldConfig, *args, **kwargs):
        super().__init__(shape=config.shape, *args, **kwargs)

        self.action_space = spaces.Discrete(6)
        self.action_space_1 = spaces.Discrete(6)
        self.config = config
        self.default_inventory = Counter(
            {inv_type.name: inv_type.default_quantity for inv_type in self.config.inventory})
        self.rand = Random()

        """
        Up: 0,
        Right:1,
        Down: 2,
        Left: 3,
        No-op: 4,
        Tile action: 5"""

        self.done = True
        #self.position = list()
        self.position: Tuple[int, int] = (0, 0)
        self.position_1: Tuple[int, int] = (0, 0)
        self.special_tiles: Dict[Tuple[int, int], MineWorldTileType] = dict()
        self.inventory = Counter()
        self.turnn = 1

    def get_turnn(self):
        return self.turnn

    def step(self, action: int):
        # this functoin only updates movement and inventory in the env
        if self.turnn == 1:
            self.turnn = 2
            assert self.action_space.contains(action)
            assert not self.done

            atomic_propositions = set()

            if action < 5:
                # Movement or no-op
                action_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
                new_place = element_add(self.position, action_offsets[action])

                can_move = self._in_bounds(new_place)

                if new_place in self.special_tiles:
                    if self.special_tiles[new_place].wall:
                        can_move = False

                if can_move:
                    self.position = new_place
            else:
                if self.position in self.special_tiles:
                    this_tile: MineWorldTileType = self.special_tiles[self.position]
                    try:
                        new_inv = this_tile.apply_inventory(self.inventory)
                        for inv_config in self.config.inventory:
                            if new_inv[inv_config.name] > inv_config.capacity:
                                new_inv[inv_config.name] = inv_config.capacity
                        self.inventory = new_inv
                        atomic_propositions.add(this_tile.ap_name)
                        if this_tile.consumable:
                            del self.special_tiles[self.position]
                    except ValueError:  # Couldn't apply inventory
                        pass
        else:
            self.turnn = 1
            assert self.action_space.contains(action)
            assert not self.done

            atomic_propositions = set()

            if action < 5:
                # Movement or no-op
                action_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
                new_place = element_add(self.position_1, action_offsets[action])

                can_move = self._in_bounds(new_place)

                if new_place in self.special_tiles:
                    if self.special_tiles[new_place].wall:
                        can_move = False

                if can_move:
                    self.position_1 = new_place
            # else:
            #     if self.position_1 in self.special_tiles:
            #         this_tile: MineWorldTileType = self.special_tiles[self.position_1]
            #         try:
            #             new_inv = this_tile.apply_inventory(self.inventory)
            #             for inv_config in self.config.inventory:
            #                 if new_inv[inv_config.name] > inv_config.capacity:
            #                     new_inv[inv_config.name] = inv_config.capacity
            #             self.inventory = new_inv
            #             atomic_propositions.add(this_tile.ap_name)
            #             if this_tile.consumable:
            #                 del self.special_tiles[self.position_1]
            #         except ValueError:  # Couldn't apply inventory
            #             pass
        info = {
            'atomic_propositions': atomic_propositions,
            'inventory': self.inventory.copy()
        }

        # Reward is always 0 because it's minecraft. Exploration is reward in itself, and the only limit is imagination
        return self._get_observation(), 0, self.done, info

    def seed(self, seed=None):
        self.rand.seed(seed)

    def reset(self):
        self.done = False
        self.turnn = 1
        self.position = self.config.initial_position
        self.position_1 = self.config.initial_position_1

        if not self.position:
            self.position = self.rand.randrange(0, self.shape[0]), self.rand.randrange(0, self.shape[1])
        self.inventory = self.default_inventory.copy()
        self.special_tiles = self._get_tile_positioning()

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

    def _get_observation(self):
        # env returning the state to the MCTS
        tiles = tuple(
            frozenset(space for space, content in self.special_tiles.items() if content is placement.tile) for
            placement in self.config.placements)

        inv_ratios = tuple(
            self.inventory[inv_config.name] / inv_config.capacity for inv_config in self.config.inventory)

        return (
            self.position,
            tiles,
            inv_ratios,
            self.position_1,
            self.turnn
        )

    def render(self, mode='human'):
        def render_func(x, y):
            if self.position  == (x, y) and self.position_1 == (x, y):
                agent_str = "X"
            elif self.position  == (x, y):
                agent_str = "1"
            elif self.position_1 == (x, y):
                agent_str = "2"
            else:
                agent_str = " "

            # agent_str   = "1" if self.position   == (x, y) else " "
            tile_str = self.special_tiles[(x, y)].grid_letter if (x, y) in self.special_tiles else " "

            return agent_str + tile_str,False, False

        print(self._render(render_func, 2), end="")
        print(dict(self.inventory))

    def save_state(self):
        return self.position,self.position_1,self.turnn, self.done, self.special_tiles.copy(), self.inventory.copy()

    def load_state(self, state):
        self.position, self.position_1, self.turnn, self.done, spec_tile, inv = state
        self.special_tiles = spec_tile.copy()
        self.inventory = inv.copy()
