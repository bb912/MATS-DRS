from abc import ABCMeta, abstractmethod
from typing import Callable, TypeVar, Generic, List

import torch
from torch.optim import Adam

from autograph.lib.util import one_hot_multi
from autograph.net.curiosity.curiosity_module import CuriosityModule

T = TypeVar("T")


class CuriosityOptimizer(metaclass=ABCMeta):

    @abstractmethod
    def get_curiosity(self, states, actions, next_states):
        pass

    @abstractmethod
    def train(self, states, actions, next_states, train_rounds):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, dict):
        pass


class ModuleCuriosityOptimizer(Generic[T], CuriosityOptimizer):
    def __init__(self, curiosity: CuriosityModule, state_transformer: Callable[[T], torch.Tensor], num_actions: int,
                 lr: float, device):
        self.curiosity = curiosity
        self.optim = Adam(curiosity.parameters(), lr=lr)
        self.num_actions = num_actions
        self.state_transformer = state_transformer
        self.device = device

    def transform_sas(self, states: List[T], actions: List[int], next_states: List[T]):
        s_t = None
        a_t = None
        ns_t = None
        if states:
            s_t = torch.stack(tuple(self.state_transformer(state) for state in states), dim=0).to(self.device)

        if actions:
            actions_tensor_scalar = torch.as_tensor(actions).to(self.device)
            a_t = one_hot_multi(actions_tensor_scalar, self.num_actions, self.device).float()

        if next_states:
            ns_t = torch.stack(tuple(self.state_transformer(ns) for ns in next_states), dim=0).to(self.device)

        return s_t, a_t, ns_t

    def get_curiosity(self, states: List[T], actions: List[int], next_states: List[T]):
        s_t, a_t, ns_t = self.transform_sas(states, actions, next_states)
        return self.get_curiosity_preprocessed(s_t, a_t, ns_t)

    def get_curiosity_preprocessed(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor):
        return self.curiosity.get_intrinsic_reward(states, actions, next_states)

    def train(self, states: List[T], actions: List[int], next_states: List[T], train_rounds=1):
        s_t, a_t, ns_t = self.transform_sas(states, actions, next_states)
        return self.train_preprocessed(s_t, a_t, ns_t, train_rounds)

    def train_preprocessed(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor,
                           train_rounds: int = 1):

        total_train_loss = 0

        for i in range(train_rounds):
            self.optim.zero_grad()
            train_loss = self.curiosity.get_training_loss(states, actions, next_states).mean()
            train_loss.backward()
            self.optim.step()

            total_train_loss += float(train_loss)

        return total_train_loss / train_rounds if train_rounds > 0 else 0

    def state_dict(self):
        return {
            "optim": self.optim.state_dict()
        }

    def load_state_dict(self, dict):
        if dict and dict.get("optim"):
            self.optim.load_state_dict(dict["optim"])


class NoopCuriosityOptimizer(CuriosityOptimizer):

    def get_curiosity(self, states, actions, next_states):
        return torch.zeros(len(states), dtype=torch.float)

    def train(self, states, actions, next_states, train_rounds):
        return 0

    def state_dict(self):
        return {}

    def load_state_dict(self, dict):
        pass
