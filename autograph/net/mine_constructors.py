import functools

from autograph.lib.envs.mineworldenv_team import MineWorldEnv
from autograph.net.curiosity.rnd_models import RND
from autograph.net.mazenet import Mazenet
from autograph.net.minenet import Minenet, minecraft_obs_rewrite


def num_channels(env: MineWorldEnv):

    sum = 1 + len(env.config.placements)
    for i in range(0, len(env.config.inventories)):
        sum += len(env.config.inventories[i])
    return sum

def num_channels_turn_inventory(env: MineWorldEnv):

    sum = 2 + len(env.config.placements)
    for i in range(0, len(env.config.inventories)):
        sum += len(env.config.inventories[i])
    return sum

def num_channels_turn_dirs_or_cones(env: MineWorldEnv):

    sum = 2 + len(env.config.placements)
    for i in range(0, len(env.config.inventories)):
        sum += len(env.config.inventories[i])
        sum += 1
    return sum

def num_channels_turn_dirs_and_cones(env: MineWorldEnv):

    sum = 2 + len(env.config.placements)
    for i in range(0, len(env.config.inventories)):
        sum += len(env.config.inventories[i])
        sum += 2
    return sum

def num_channels_pos_cones_invs(env: MineWorldEnv):

    # turn + num_special tiles
    sum = 1 + len(env.config.placements)
    sum += len(env.config.init_positions) #for just cones
    #add inventories
    for i in range(0, len(env.config.inventories)):
        sum += len(env.config.inventories[i])
        # cone_view + position

    return sum

def minenet_v1(env: MineWorldEnv, num_blocks, **kwargs):
    return Minenet(env.shape, num_channels(env), num_blocks, env.action_space.n, **kwargs)


def mine_mazenet_v1(env: MineWorldEnv):
    return Mazenet(env.shape, actions_n=env.action_space.n, in_channels=num_channels_turn_inventory(env), initstride=1, initpadding=2)

def mine_mazenet_dirs(env: MineWorldEnv):
    #return Mazenet(env.shape, actions_n=env.action_space.n, in_channels=num_channels_turn_dirs_or_cones(env), initstride=1, initpadding=2)
    return Mazenet(env.shape, actions_n=env.action_space.n, in_channels=num_channels_pos_cones_invs(env), initstride=1, initpadding=2)


# inputs for both inventories, turn, and both positions
def mine_mazenet_inv(env: MineWorldEnv):
    return Mazenet(env.shape, actions_n=env.action_space.n, in_channels=num_channels_turn_inventory(env), initstride=1, initpadding=2)


def minernd_v1(env: MineWorldEnv, feature_space: int):
    return RND(num_channels(env), env.shape, feature_space, init_stride=1)


def mine_obs_rewriter_creator(env: MineWorldEnv):
    return functools.partial(minecraft_obs_rewrite, env.shape)
