import gym
import torch
from torch.nn import Module

from autograph.net.basic_net import BasicNet
from autograph.net.curiosity.no_op import NoopCuriosity


def gym_make(dict):
    return gym.make(**dict)


def no_op_cur_make(env):
    return NoopCuriosity()


class NoopNet(Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.dummy_param = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, input: torch.Tensor):
        batch = input.shape[0]

        return torch.full(size=(batch, self.num_actions), fill_value=(1.0 / self.num_actions), device=input.device,
                          requires_grad=True), torch.zeros(size=(batch, 1), device=input.device, requires_grad=True)


def no_op_make(env):
    return NoopNet(env.action_space.n)


def basic_net(env: gym.Env, intermediate_size):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)

    assert len(env.observation_space.shape) == 1

    return BasicNet(input_size=env.observation_space.shape[0],
                    intermediate_size=intermediate_size,
                    output_size=env.action_space.n)
