import functools
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from autograph.lib.envs.mazeenv import n_hot_grid
from autograph.lib.util import const_plane
from autograph.net.curiosity.utils import Flatten


class Residual(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input):
        output = self.inner(input)
        return input + output


class Scalarize(nn.Module):
    def forward(self, input: Tensor):
        return input.view((input.size(0),))


KERNEL_SIZE = 3
PADDING_AMOUNT = 1


class Minenet(nn.Module):
    def __init__(self, shape, num_channels, num_blocks, num_actions,
                 num_intermediate_filters=32,
                 num_policy_head_filters=2,
                 num_value_head_filters=1,
                 disable_value=False,
                 disable_policy=False,
                 separate_networks=False):
        super().__init__()

        grid_size = 1
        for dim in shape:
            grid_size *= dim

        self.disable_value = disable_value
        self.disable_policy = disable_policy
        self.num_actions = num_actions
        self.separate_networks = separate_networks

        # Basically the architecture from AlphaGo
        def generate_common():
            init_conv = nn.Sequential(
                nn.Conv2d(num_channels, num_intermediate_filters, kernel_size=KERNEL_SIZE, padding=PADDING_AMOUNT),
                nn.BatchNorm2d(num_intermediate_filters),
                nn.LeakyReLU()
            )

            blocks = [nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(in_channels=num_intermediate_filters,
                                  out_channels=num_intermediate_filters,
                                  kernel_size=KERNEL_SIZE,
                                  padding=PADDING_AMOUNT),
                        nn.BatchNorm2d(num_intermediate_filters),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=num_intermediate_filters,
                                  out_channels=num_intermediate_filters,
                                  kernel_size=KERNEL_SIZE,
                                  padding=PADDING_AMOUNT),
                        nn.BatchNorm2d(num_intermediate_filters))
                ),
                nn.LeakyReLU()
            ) for _ in range(num_blocks)]

            return nn.Sequential(
                init_conv, *blocks
            )

        self.policy_trunk = generate_common()
        if separate_networks:
            self.value_trunk = generate_common()

        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=num_intermediate_filters,
                      out_channels=num_policy_head_filters,
                      kernel_size=1),
            nn.BatchNorm2d(num_policy_head_filters),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(grid_size * num_policy_head_filters, num_actions)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=num_intermediate_filters,
                      out_channels=num_value_head_filters,
                      kernel_size=1),
            nn.BatchNorm2d(num_value_head_filters),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(grid_size * num_value_head_filters, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            # Scalarize(),
            nn.Tanh()
        )

    def forward(self, x: Tensor):
        policy_trunk_out = self.policy_trunk(x)

        if self.separate_networks:
            value_trunk_out = self.value_trunk(x)
        else:
            value_trunk_out = policy_trunk_out

        batch = x.size(0)
        if self.disable_policy:
            pol = torch.full(size=(batch, self.num_actions), fill_value=1.0 / self.num_actions, device=x.device,
                             requires_grad=True)
        else:
            pol = self.policy_head(policy_trunk_out)

        if self.disable_value:
            val = torch.zeros(size=(batch, 1), device=x.device, requires_grad=True)
        else:
            val = self.value_head(value_trunk_out)

        return pol, val


@functools.lru_cache(16384)
def minecraft_obs_rewrite_original(shape, obs):
    #position, tile_locs, inventories= obs
    position, tile_locs, inventories,*_ = obs
    # Convert to float?
    position_tile_layers = tuple(torch.from_numpy(n_hot_grid(shape, layer)).float() for layer in (position, *tile_locs))
    inventory_layers = tuple(const_plane(shape, layer).float() for layer in inventories)
    return torch.stack((*position_tile_layers, *inventory_layers), dim=0)


@functools.lru_cache(16384)
def minecraft_obs_rewrite_no_dirs(shape, obs):
    # position, tile_locs, inventories= obs
    position, tile_locs, inventories, turn = obs
    posLayers = list(tuple(torch.from_numpy(n_hot_grid(shape, layer)).float() for layer in (position, *tile_locs)))

    # posLayers = torch.stack(list(tuple(torch.from_numpy(n_hot_grid(shape, layer)).float() for layer in (position, *tile_locs))), dim=2)
    # pos = np.array(posLayers)
    # Convert to float?
    # position_tile_layers = tuple(torch.from_numpy(pos))
    inventory_layers = list()
    turnn = list()
    for inventory in inventories:
        for layer in inventory:
            inventory_layers.append(const_plane(shape, layer))
    turnn.append(const_plane(shape, turn))


    x = 1

    # inv_list = list(tuple(torch.from_numpy(np.ndarray(list((const_plane(shape, layer)) for layer in inv))).float() for inv in inventories))

    # inventory_layers = tuple(torch.from_numpy(nparr).float())
    # inv_list = tuple(np.array(tuple(const_plane(shape, layer).float() for layer in inv)) for inv in inventories)
    # nparr = np.array([np.array(xi) for xi in inv_list])
    return torch.stack((*posLayers, *inventory_layers, *turnn), dim=0)


@functools.lru_cache(16384)
def minecraft_obs_rewrite_for_seeBoth(shape, obs):
    #position, tile_locs, inventories= obs
    if len(obs) == 4:
        stop = 0
    position, tile_locs, inventories, turn, full_cone_views = obs
    posLayers = list(tuple(torch.from_numpy(n_hot_grid(shape, layer)).float()
                           for layer in (position, *tile_locs, *full_cone_views)))

    #posLayers = torch.stack(list(tuple(torch.from_numpy(n_hot_grid(shape, layer)).float() for layer in (position, *tile_locs))), dim=2)
    #pos = np.array(posLayers)
    # Convert to float?
    #position_tile_layers = tuple(torch.from_numpy(pos))
    inventory_layers = list()
    turnn = list()
    for inventory in inventories:
        for layer in inventory:
            inventory_layers.append(const_plane(shape, layer))
    turnn.append(const_plane(shape, turn))

    #dirs = list()
    #for dir in directions:
    #    dirs.append(const_plane(shape, dir))
    x = 1



    #inv_list = list(tuple(torch.from_numpy(np.ndarray(list((const_plane(shape, layer)) for layer in inv))).float() for inv in inventories))

    #inventory_layers = tuple(torch.from_numpy(nparr).float())
    #inv_list = tuple(np.array(tuple(const_plane(shape, layer).float() for layer in inv)) for inv in inventories)
    #nparr = np.array([np.array(xi) for xi in inv_list])
    return torch.stack((*posLayers, *inventory_layers, *turnn), dim=0)


    #return torch.stack((*posLayers, *inventory_layers, turn), dim=0)

@functools.lru_cache(16384)
def minecraft_obs_rewrite(shape, obs):
    #position, tile_locs, inventories= obs

    positions, tile_locs, inventories, turn, full_cone_views = obs

    #all_positions = list(tuple(torch.from_numpy(n_hot_grid(shape, position)).float()
    #                           for position in positions))

    special_tiles = list(tuple(torch.from_numpy(n_hot_grid(shape, layer)).float()
                               for layer in tile_locs))

    full_cone_view = list(tuple(torch.from_numpy(n_hot_grid(shape, layer)).float()
                                for layer in full_cone_views))

    inventory_layers = list()
    turnn = list()
    for inventory in inventories:
        for layer in inventory:
            inventory_layers.append(const_plane(shape, layer))
    turnn.append(const_plane(shape, turn))

    return torch.stack((*special_tiles, *full_cone_view, *inventory_layers, *turnn), dim=0)

