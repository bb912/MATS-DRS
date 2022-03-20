import functools

from autograph.lib.envs.mazeenv import FuelMazeEnv, transform_coordinate
from autograph.net.curiosity.rnd_models import RND
from autograph.net.mazenet import Mazenet, maze_obs_rewrite


def mazenet_v1(env: FuelMazeEnv):
    return Mazenet(transform_coordinate(env.shape), env.action_space.n)


def mazernd_v1(env: FuelMazeEnv, feature_space: int):
    return RND(7, transform_coordinate(env.shape), feature_space)


def maze_obs_rewrite_creator(env: FuelMazeEnv):
    return functools.partial(maze_obs_rewrite, transform_coordinate(env.shape))
