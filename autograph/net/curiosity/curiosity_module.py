import abc

from torch.nn import Module


class CuriosityModule(abc.ABC, Module):

    def __init__(self):
        super().__init__()
        # self.get_single_intrinsic_reward = single_batch(self.get_intrinsic_reward)
        # self.get_single_training_loss = single_batch(self.get_training_loss)

    @abc.abstractmethod
    def get_intrinsic_reward(self, state, action, next_state):
        raise NotImplemented

    @abc.abstractmethod
    def get_training_loss(self, state, action, next_state):
        raise NotImplemented

    # def get_single_intrinsic_reward(self, state, action, next_state):
    #    raise NotImplemented

    # def get_single_training_loss(self, state, action, next_state):
    #    raise NotImplemented
