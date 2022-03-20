import torch
from torch import Tensor

from autograph.net.curiosity.curiosity_module import CuriosityModule


class NoopCuriosity(CuriosityModule):
    def get_intrinsic_reward(self, state: Tensor, action, next_state):
        return torch.zeros(size=(state.size(0),), device=state.device)

    def get_training_loss(self, state, action, next_state):
        return torch.zeros(size=(state.size(0),), device=state.device, requires_grad=True)

    def forward(self, input: Tensor):
        return torch.zeros(size=(input.size(0),), device=input.device, requires_grad=True)
