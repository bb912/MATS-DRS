import functools

import torch
import torch.nn as nn
from torch import tensor

from autograph.lib.envs.mazeenv import n_hot_grid
from autograph.lib.util import const_plane


class Mazenet(nn.Module):
    def __init__(self, maze_shape, actions_n, in_channels=7, initstride=2, initpadding=1):
        super(Mazenet, self).__init__()

        self.maze_shape = maze_shape

        self.maze_size = 1
        for dim in maze_shape:
            self.maze_size *= dim

        self.net_common_before_flatten = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(5, 5), stride=initstride,
                      padding=initpadding).float(),
            # n*trans(x)*trans(y) -> 32*x*y
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=2).float(),
            nn.ELU()
        )

        output_before_flatten: tensor = self.net_common_before_flatten(torch.zeros(1, in_channels, *maze_shape))
        lin_layer_size = output_before_flatten.flatten(1).shape[1]

        self.net_common_after_flatten = nn.Sequential(
            nn.Linear(lin_layer_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.net_policy = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions_n),
        )

        self.net_value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        common: tensor = self.net_common_before_flatten(x)
        common_flat = common.flatten(1)
        common_after_linear = self.net_common_after_flatten(common_flat)
        val = self.net_value(common_after_linear)
        pol = self.net_policy(common_after_linear)

        return pol, val

    def batch_states(self, states):
        processed = torch.stack(tuple(self.rewrite_obs(s) for s in states))
        return self(processed)

    def _process_single_tensor(self, tens):
        return tens.float()

    def forward_obs(self, x, device):
        info = maze_obs_rewrite(self.maze_size, x)
        pols, vals = self(info.to(device).unsqueeze(0))
        return pols[0].cpu(), vals[0].cpu()


@functools.lru_cache(16384)
def maze_obs_rewrite(shape, obs):
    fuel_level = const_plane(shape, obs[0])
    others = tuple(torch.from_numpy(n_hot_grid(shape, layer)).float() for layer in obs[1:])
    return torch.stack((fuel_level, *others), dim=0).float()
