from abc import ABC, abstractmethod

from gym import Env


class SaveLoadEnv(Env, ABC):
    """
    The state that is saved only needs to be the properties of the environment that change during a run. Any properties
    that change between episodes don't need to be saved and loaded
    """

    @abstractmethod
    def save_state(self):
        """
        :return: An object that can be used to represent the current state
        """
        raise NotImplemented

    @abstractmethod
    def load_state(self, state):
        """
        Restore the state of the environment
        """
        raise NotImplemented
