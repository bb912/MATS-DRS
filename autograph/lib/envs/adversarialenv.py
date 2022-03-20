from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Tuple, Any, Sequence

from gym import Env


class PlayerID(IntEnum):
    Player1 = 1
    Player2 = -1


class AdversarialEnv(Env, ABC):

    @abstractmethod
    def get_current_player(self) -> PlayerID:
        """What player is allowed to make a move right now"""

        raise NotImplemented

    @abstractmethod
    def step(self, action: Tuple[PlayerID, Any]):
        """Have a certain player take a step in the environment

        All rewards are from the perspective of the player that just took a step. This can potentially include rewards
        that accumulated for this player over the course of other players' turns."""

        raise NotImplemented

    @abstractmethod
    def get_allowed_moves(self) -> Sequence[bool]:
        """Returns a bitmap of allowed moves from the current state."""

        raise NotImplemented
