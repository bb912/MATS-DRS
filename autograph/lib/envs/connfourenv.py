from typing import Tuple, Union

import gym.spaces as spaces
import numpy as np
from gym.spaces import Box

from autograph.lib.envs.adversarialenv import AdversarialEnv, PlayerID
from autograph.lib.envs.saveloadenv import SaveLoadEnv
from autograph.lib.util import element_add, element_neg

PATTERNS = ((0, 1), (1, 0), (1, 1), (1, -1))


class ConnectFourEnv(SaveLoadEnv, AdversarialEnv):
    def __init__(self, board_shape: Tuple[int, int] = (7, 6), connect_n=4, from_perspective_of_player: bool = False):

        self.action_space = spaces.Discrete(board_shape[0])

        # Observation is always from the perspective of the current player (the player that will take a turn next)
        # And the player that did not go last after the game ends
        self.observation_space = Box(-1, 1, board_shape, dtype=np.int)

        self.from_perspective_of_player = from_perspective_of_player
        self.connect_n = connect_n
        self.board_shape: Tuple[int, int] = board_shape

        self.done: bool = None
        self.current_player: PlayerID = None
        self.board: np.ndarray = None
        self.next_slots: np.ndarray = None

        self.reset()

    def save_state(self):
        return self.current_player, self.board.copy(), self.done, self.next_slots.copy()

    def load_state(self, state):
        self.current_player, board_temp, done, ns_temp = state
        self.board = board_temp.copy()
        self.next_slots = ns_temp.copy()

    def get_current_player(self) -> PlayerID:
        return self.current_player

    def step(self, action: Union[int, Tuple[PlayerID, int]]):
        assert not self.done

        if isinstance(action, tuple):
            player, slot = action
            assert player == self.current_player
        else:
            player = self.current_player
            slot = action

        assert 0 <= slot < self.board_shape[0]

        row = self.next_slots[slot]
        assert 0 <= row < self.board_shape[1]
        assert not self.board[slot][row]

        self.board[slot][row] = player

        self.next_slots[slot] = row + 1

        self.current_player *= -1

        if self.is_game_over((slot, row)):
            self.done = True
            return self._observation(), 1, True, {}
        else:
            return self._observation(), 0, False, {}

    def in_bounds(self, point):
        return 0 <= point[0] < self.board_shape[0] and 0 <= point[1] < self.board_shape[1]

    def continuous_in_direction(self, point, direction, value):
        """How many spaces in the given direction from the point are the spaces equal to the given value?

        Does not count the point that we started on"""
        count = 0
        point = element_add(point, direction)

        while self.in_bounds(point) and self.board[point] == value:
            count += 1
            point = element_add(point, direction)

        return count

    def is_game_over(self, starting_point):
        player = self.board[starting_point]
        assert player != 0

        for pattern in PATTERNS:

            num_in_positive = self.continuous_in_direction(starting_point, pattern, player)

            neg_pattern = element_neg(pattern)
            num_in_negative = self.continuous_in_direction(starting_point, neg_pattern, player)

            num_continuous_pieces = 1 + num_in_positive + num_in_negative  # Also count the starting point

            if num_continuous_pieces >= self.connect_n:
                return True

        return False

    def get_allowed_moves(self):
        return self.next_slots < self.board_shape[1]

    def reset(self):
        self.done = False
        self.current_player = 1
        self.board = np.zeros(shape=self.board_shape, dtype=np.int8)
        self.next_slots = np.zeros(shape=(self.board_shape[0],), dtype=np.int8)

        return self._observation()

    def _observation(self):
        if self.from_perspective_of_player:
            return self.board.copy() * self.current_player
        else:
            return self.board.copy()

    def render(self, mode='human'):
        width = self.board_shape[0]
        horizsep = "--".join(["+"] * (width + 1)) + "\n"

        output = horizsep

        for row in self.board.transpose()[::-1]:
            for col in row:
                output += "| "

                if col == PlayerID.Player1:
                    output += "X"
                elif col == PlayerID.Player2:
                    output += "O"
                else:
                    output += " "

            output += "|\n"

            output += horizsep

        print(output)
