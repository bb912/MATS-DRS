import math
import random
import sys
from collections import Counter, namedtuple
from typing import Tuple, List, Union, Iterable, FrozenSet

import gym.spaces as spc
import networkx
import numpy as np
from gym.spaces import Discrete

import autograph.lib.envs.saveloadenv as sl
from autograph.lib.envs.gridenv import GridEnv
from autograph.lib.util import element_add
from autograph.lib.envs.mineworldenv_adv_list import MineWorldAgent

FuelMazeObservation = namedtuple("FuelMazeObservation", ["fuel_level", "position", "previous_position", "walls", "keys",
                                                         "fuel_positions", "goal"])


def transform_coordinate(coord, offset=(0, 0)):
    return element_add(offset, tuple(c * 2 + 1 for c in coord))


def n_hot_gridA(shape: Tuple[int, int], grid_positions: Union[None, Tuple[int, int], Iterable[Tuple[int, int]]],
               grid=None):
    if grid is None:
        grid = np.zeros(shape, dtype=np.uint8)

    if grid_positions is None:
        grid_positions = []

    if isinstance(grid_positions, tuple) and len(grid_positions) == 2 and isinstance(grid_positions[0], int):
        grid_positions = [grid_positions]

    for pos in grid_positions:
        grid[pos] = 1

    return grid

def n_hot_grid(shape: Tuple[int, int], grid_positions: Union[None, Tuple[MineWorldAgent, MineWorldAgent]],
               grid=None):
    if grid is None:
        grid = np.zeros(shape, dtype=np.float_)

    if grid_positions is None:
        grid_positions = []

    if isinstance(grid_positions, tuple):
        if grid_positions == (-10, -10):
            pos = (0, 0)
            grid[pos] = 0
        else:
            grid[grid_positions] = 1
        return grid

    for pos in grid_positions:
        if pos == (-10, -10):
            pos = (0, 0)
            grid[pos] = 0
        else:
            grid[pos] = 1

    return grid

def tupleize(arr: np.ndarray):
    if len(arr.shape) <= 1:
        return tuple(arr)
    else:
        return map(tupleize, arr)


class SetSpace(spc.Space):
    def __init__(self, inner: spc.Space, minsize: int, maxsize: int):
        super().__init__()
        self.maxsize = maxsize
        self.minsize = minsize
        self.inner = inner
        self.rand: random.Random = None

        self.seed(random.randint(0, 10000000))

    def sample(self):
        num = self.rand.randint(self.minsize, self.maxsize)
        return frozenset({self.inner.sample() for _ in range(num)})

    def seed(self, seed):
        self.rand = random.Random(seed)
        self.inner.seed(self.rand.randint(0, 10000000))

    def contains(self, x: FrozenSet):
        return all(map(self.inner.contains, x))


class FuelMazeEnv(sl.SaveLoadEnv, GridEnv):
    """
    An OpenAI Gym environment that represents a maze with fuel and keys
    """

    @staticmethod
    def from_dict(dict):
        return FuelMazeEnv(shape=tuple(dict["shape"]),
                           num_keys=dict["keys"],
                           max_fuel_level=dict["fuel_cap"],
                           max_fuel_dist=dict["max_fuel_dist"],
                           loop_factor=dict["loop_factor"],
                           seed=dict["random_seed"],
                           randomize_on_reset=dict["random_on_reset"])

    def __init__(self, shape=(10, 10), num_keys=1, max_fuel_level=20, max_fuel_dist=10, loop_factor=0.05, seed=None,
                 randomize_on_reset=False, *args, **kwargs):
        """
        Construct a new instance of this environment.
        :param shape: A 2-tuple (width, height) of the size of the environment
        :param num_keys: How many keys the maze has
        :param max_fuel_level: The maximum fuel capacity of the agent
        :param max_fuel_dist: The farthest away any given space is from a fuel space
        :param loop_factor: The fraction of walls to remove after generating the maze. 0 is a perfect maze, 1 is a grid
        :param seed: A RNG seed
        :param randomize_on_reset: Whether to randomize maze parameters when it is reset
        """

        super(FuelMazeEnv, self).__init__(shape=shape, *args, **kwargs)

        num_spaces = shape[0] * shape[1]

        trans_shape = transform_coordinate(shape)

        pos_space = spc.Tuple((Discrete(trans_shape[0]), Discrete(trans_shape[1])))
        num_trans_spaces = trans_shape[0] * trans_shape[1]

        # Closest fuel can be placed together is max_fuel_dist + 1 from each other in "easiest" case
        max_fuel_num = math.ceil(num_spaces / (max_fuel_dist + 1))

        self.observation_space = spc.Tuple((Discrete(max_fuel_level),
                                            pos_space,
                                            SetSpace(pos_space, 0, 1),
                                            SetSpace(pos_space, 0, num_trans_spaces),
                                            SetSpace(pos_space, 0, num_keys),
                                            SetSpace(pos_space, 0, max_fuel_num),
                                            pos_space))
        """
        Tuple:
        (fuel, current position, previous position, wall positions, key positions, fuel positions, goal position)  
        
        Every corner, edge, and cell has a unique position in the observation space:
        (0,0)
        ↓    ↙(3,0)
        +--+--+
        |11|31|
        +--+--+ ←(4, 2)
        |13|33|
        +--+--+ ←(4, 4) 
        """

        self.action_space = spc.Discrete(4)
        """
        0: up
        1: right
        2: down
        3: left
        4: no-op (this might only be used if an arbitrary LTL spec calls for it)
        """

        seed = seed if seed else random.randint(0, sys.maxsize)

        self.rand = random.Random(seed)  #: rng instance
        self.num_keys = num_keys  #: how many keys in maze
        self.max_fuel_dist = max_fuel_dist  #: max distance from any tile to a fuel tile
        self.start_fuel_level = max_fuel_level  #: max capacity of agent, also starting fuel amount
        self.goal = element_add(shape, (-1, -1))  #: try to get to bottom-right
        self.keys = []  #: locations of all keys
        self.fuel = []  #: locations of all fuel
        self.loop_factor = loop_factor  #: how many edges to remove after creation
        self.maze: networkx.Graph = None  #: the maze graph itself
        self.edge_grid = None  #: Cached locations of edges in observation format
        self.randomize_on_reset = randomize_on_reset

        self.current_fuel_level = None  #: how much fuel the robot currently has, -1 per time step
        self.last_position = None  #: (x, y) of agent's previous location
        self.keys_left: List = None  #: locations of remaining keys
        self.seen = Counter()
        self.done = True  #: true if episode is done
        self.position = None  #: (x, y) of agent's location

        self._generate_maze()

    def step(self, action: int):
        """
        Takes one step through the maze
        :param action: The action to take- see action_space for details
        :return: observation, reward, done, empty dict
        """
        assert self.action_space.contains(action)
        assert not self.done

        action_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]

        self.last_position = self.position

        next_place = self.position
        if self._exists_offset(self.position, action_offsets[action]):
            next_place = element_add(self.position, action_offsets[action])

        if next_place in self.keys_left:
            self.keys_left.remove(next_place)

        if next_place in self.fuel:
            self.current_fuel_level = self.start_fuel_level
        else:
            self.current_fuel_level -= 1

        self.position = next_place

        # reward = sum(self.position) - sum(self.last_position)
        reward = 0

        if self.position == self.goal and len(self.keys_left) == 0:
            reward = 1
            self.done = True
        elif self.current_fuel_level == 0:
            reward = 0
            self.done = True

        return self._get_observation(), reward, self.done, self._get_info()

    def reset(self):
        """
        Reset state of the simulation
        :return: The initial observation
        """
        # maxx, maxy = self.shape
        # self.position = (self.rand.randint(0, maxx - 1), self.rand.randint(0, maxy - 1))

        if self.randomize_on_reset:
            self._generate_maze()

        self.position = (0, 0)
        self.last_position = None
        self.keys_left = self.keys[:]
        self.current_fuel_level = self.start_fuel_level
        self.done = False
        self.seen.clear()

        return self._get_observation()

    def render(self, mode='human'):
        """
        Print out the state of the simulation
        """

        def render_func(x, y):
            space = x, y
            return self._get_space_inside(space), \
                   not self._exists_offset(space, (0, -1)), \
                   not self._exists_offset(space, (-1, 0))

        out = sys.stdout
        maze = self._render(render_func, 2)
        out.write(maze)
        out.write("Fuel left: %s\n" % (self.current_fuel_level,))

    def seed(self, seed=None):
        """
        Change the seed of the simulation
        """
        seed = seed if seed else random.randint(0, sys.maxsize)
        self.rand = random.Random(seed)
        return seed

    def _edge_grid(self):
        gridpos = set()

        # Fill in edges that lead out of bounds or that don't have a connection
        for pos in np.ndindex(self.shape):
            for offset in ((-1, 0), (0, -1), (1, 0), (0, 1)):
                other_one = element_add(pos, offset)
                if not (self._in_bounds(other_one) and self.maze.has_edge(pos, other_one)):
                    gridpos.add(transform_coordinate(pos, offset))

        # Fill in corners
        for pos in np.ndindex(element_add(self.shape, (1, 1))):
            gridpos.add(transform_coordinate(pos, (-1, -1)))

        return frozenset(gridpos)

    def _get_observation(self):
        """
        Counts where we currently are
        :return: An observation of the maze
        """
        self.seen[self.position] += 1

        return FuelMazeObservation(
            self.current_fuel_level,
            transform_coordinate(self.position),
            frozenset({transform_coordinate(self.last_position)}) if self.last_position else frozenset(),
            self.edge_grid,
            frozenset(map(transform_coordinate, self.keys_left)),
            frozenset(map(transform_coordinate, self.fuel)),
            transform_coordinate(self.goal)
        )

    def _get_info(self):
        return {
            "max_fuel": self.start_fuel_level,
            "maze_shape": self.shape
        }

    def _get_space_inside(self, space):
        """
        For rendering: is the agent in a given location, and is any location a goal, fuel, or key
        """
        out = ""

        if self.position == space:
            out += "A"
        elif space in self.seen:
            out += "*"
        else:
            out += " "

        if space == self.goal:
            out += "G"
        elif space in self.keys_left:
            out += "K"
        elif space in self.fuel:
            out += "F"
        else:
            out += " "

        return out

    def _generate_maze(self):
        """
        Generate a maze, along with keys and fuel positions
        """
        net: networkx.Graph = networkx.Graph()

        # Initialize a grid of nodes, each connected to four neighbors
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                place = (x, y)
                net.add_node(place)

                for neighbor in self._neighbors(place):  # Only edges down and right- no double count
                    net.add_edge(place, neighbor, weight=self.rand.random())  # Weight added for randomness in MST

        if self.loop_factor < 1:
            # Find MST (this is either Prim or Kruskal maze generation)
            net = networkx.algorithms.minimum_spanning_tree(net)

            # Add back in random edges according to looping factor
            for node in net.nodes:
                for neighbor in self._neighbors(node, [(1, 0), (0, 1)]):
                    if self.rand.random() < self.loop_factor:
                        if not net.has_edge(node, neighbor):
                            net.add_edge(node, neighbor)

        # Make this unweighted again
        for edge in net.edges:
            net.edges[edge]["weight"] = None

        # Add fuel until there is a fuel within n tiles of every spot
        dists_from_fuels = []
        fuels = []

        # We don't need to refuel near the goal
        dists_from_fuels.append(networkx.single_source_shortest_path_length(net, self.goal))

        # What is the farthest you need to travel from any space to find a fuel?
        # Keep if it is longer than the max
        def far_spaces():
            return [node for node in net.nodes if
                    min(dist_dict[node] for dist_dict in dists_from_fuels) > self.max_fuel_dist]

        fs = far_spaces()

        while len(fs) > 0:
            # Spawn a fuel at a random space that it far from an existing fuel
            new_fuel = self.rand.choice(fs)
            fuels.append(new_fuel)
            dists_from_fuels.append(networkx.single_source_shortest_path_length(net, new_fuel))

            fs = far_spaces()

        self.fuel = fuels

        # Generate n random keys that don't share with fuel tiles, the start, or the goal
        self.keys = self.rand.sample(
            [keyspot for keyspot in net.nodes if keyspot not in fuels and keyspot not in [self.goal, (0, 0)]],
            self.num_keys)

        self.maze = net
        self.edge_grid = self._edge_grid()

    def save_state(self):
        return self.current_fuel_level, self.position, self.last_position, self.keys_left.copy(), \
               self.seen.copy(), self.done

    def load_state(self, state):
        self.current_fuel_level, self.position, self.last_position, keys_left, seen, self.done = state
        self.keys_left = keys_left.copy()
        self.seen = seen.copy()

    def _exists_offset(self, place: Tuple[int, int], offset: Tuple[int, int]):
        """
        Can I get from a place in the maze to a relative offset
        """
        other = element_add(place, offset)
        return self._in_bounds(other) and self.maze.has_edge(place, other)
