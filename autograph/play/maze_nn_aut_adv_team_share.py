import math
import os
import signal
import sys
from typing import Callable, Any, Tuple, List, Union, Optional

import ptan
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import multiprocessing, Tensor
from torch.optim import Adam, SGD

import autograph.lib.envs.mazeenv
from autograph.lib.automata import AutomatonSet
from autograph.lib.envs.mazeenv import FuelMazeEnv, FuelMazeObservation
from autograph.lib.envs.mazeenv import transform_coordinate
from autograph.lib.envs.mineworldenv_team import MineWorldEnv
from autograph.lib.loss_functions import TakeSimilarActionsLossFunction, PPOLossFunction, \
    AdvantageActorCriticLossFunction
from autograph.lib.mcts_aut_adv_team import MCTSAutAdvList, AutStats, ExponentialAnnealedAutStats, UCBAnnealedAutStats
from autograph.lib.running import get_parallel_queue, get_parallel_queue_adv, RandomReplayTrainingLoop, \
    run_episode_generic_adv, run_episode_generic, run_episode_generic_team, run_episode_generic_team_cnn_only
from autograph.lib.shaping import AutShapingWrapperAdv
from autograph.lib.util import element_add
from autograph.lib.util.checkpoint_manager import CheckpointManager, StateDictLoadHandler, CombinedLoadHandler, \
    InitZeroLoadHandler, PickleLoadHandler, TransplantCheckpointManager
from autograph.lib.util.trace_return_step import TraceStep, TraceReturnStep
from autograph.net.curiosity.curiosity_optimizer import ModuleCuriosityOptimizer, NoopCuriosityOptimizer
from autograph.net.maze_constructors import mazenet_v1, mazernd_v1, maze_obs_rewrite_creator
from autograph.net.mine_constructors import minenet_v1, mine_obs_rewriter_creator, minernd_v1, mine_mazenet_v1, \
    mine_mazenet_inv, mine_mazenet_dirs
from autograph.net.misc_constructors import gym_make, no_op_cur_make, basic_net, no_op_make
import random

import numpy as np
from copy import deepcopy
import functools

math.sqrt(1)  # So that the import isn't optimized away (very useful when setting conditional debug breakpoints)

sys.modules["autograph.lib.mazeenv"] = autograph.lib.envs.mazeenv  # Fix broken pickle loading


def throwKeyInterr():
    raise KeyboardInterrupt()


def full_fuel(action, obs: FuelMazeObservation, rew, done, info):
    return obs.fuel_level == info["max_fuel"]


def key(action, obs: FuelMazeObservation, rew, done, info):
    return len(obs.keys) == 0


def goal(action, obs: FuelMazeObservation, rew, done, info):
    corner = element_add(info["maze_shape"], (-1, -1))
    trans_corner = transform_coordinate(corner)
    return obs.position == trans_corner


class MineInfoAutAP:
    def __init__(self, player, apname: str = None, ap_name: str = None):
        if not (apname or ap_name):
            raise ValueError("Did not provide ap_name to info aut")
        self.player = player
        self.name = apname or ap_name

    def __call__(self, action, obs, rew, done, info, cur_player):
        return cur_player == self.player and \
               self.name in info["atomic_propositions"]


class MineInventoryAP:
    def __init__(self, inventory_item, quantity, player):
        self.item = inventory_item
        self.quantity = quantity
        self.player = player

    def __call__(self, action, obs, rew, done, info, cur_player):
        return cur_player == self.player and \
               info["inventory"][self.item] == self.quantity

# player has a weapon
class MineInventoryRatioAP:
    def __init__(self, inventory_item, quantity, player):
        self.item = inventory_item
        self.quantity = quantity
        self.player = player

    def __call__(self, action, obs, rew, done, info, cur_player):
        pos, tiles, inv_ratios, turn, partial_views = obs

        return inv_ratios[self.player][2] == 1.0




class MineLocationAP:
    def __init__(self, location, player):
        self.location = tuple(location)
        self.player = player

    def __call__(self, action, obs, rew, done, info, cur_player):
        position, *_ = obs
        return position[self.player] == self.location


'''Two agents on same tile'''


class MineCatchAP:
    def __init__(self, player):
        # could be used in future to determine which players are on one tile
        self.player = player

    def __call__(self, action, obs, rew, done, info, cur_player):
        position, *_ = obs
        caught = False
        for idx in range(0, len(position)):
            if idx != self.player and idx % 2 != self.player % 2:
                caught = caught \
                         or position[self.player] == position[idx]
        return caught


class MineCatchSpecificAP:
    def __init__(self, playerCaught, playerCatching):
        # could be used in future to determine which players are on one tile
        self.playerCaught = playerCaught
        self.playerCatching = playerCatching

    def __call__(self, action, obs, rew, done, info, cur_player):
        position, *_ = obs

        return position[self.playerCaught] == position[self.playerCatching]


class MineCatchSidesAP:
    def __init__(self, playersGettingCaught, dangerSide, shape):
        # could be used in future to determine which players are on one tile
        self.playersGettingCaught = playersGettingCaught
        self.dangerSide = dangerSide
        self.shape = shape

    def __call__(self, action, obs, rew, done, info, cur_player):
        position, *_ = obs
        caught = False

        sizeX = self.shape[0]
        sizeY = self.shape[1]
        for player in range(0, len(self.playersGettingCaught)):
            for idx in range(0, len(position)):
                # Check if they are on opposing teams
                if idx != player and idx % 2 != player % 2:
                    # Check if they are on the same cell
                    if position[player][0] != -10 and position[player] == position[idx]:
                        if self.dangerSide == 'top':
                            if position[player][1] < sizeY / 2:
                                caught = True
                        else:
                            if position[player][1] >= sizeY / 2:
                                caught = True
        return caught

# finding multiple agents
class MineAllAgentsFoundAP:
    def __init__(self, agents_must_be_found):
        self.agents_must_be_found = set(agents_must_be_found)
        self.numFound = len(self.agents_must_be_found)

    def __call__(self, action, obs, rew, done, info, cur_player):
        pos, *_ = obs

        must_find_count = 0
        for idx, position in enumerate(pos):
            if idx in self.agents_must_be_found:
                if position != (-10, -10):
                    must_find_count += 1

        return must_find_count == self.numFound


# one agent found by a specific agent
class MineThisAgentFoundBy:
    def __init__(self, agent_to_find, found_by):
        self.agents_to_find = agent_to_find
        self.found_by = found_by

    def __call__(self, action, obs, rew, done, info, cur_player):
        pos, tiles, inv_ratios, turn, partial_views = obs

        finder_cone = partial_views[self.found_by]

        bool_val = pos[self.agents_to_find] in finder_cone

        return bool_val


class MineObjectFoundAP:
    def __init__(self, object_index, location):
        self.object_index = object_index
        self.location = tuple(location)

    def __call__(self, action, obs, rew, done, info, cur_player):
        pos, tiles, *_ = obs

        object_tiles = tiles[self.object_index]

        return self.location in object_tiles

# anybody on the specified team is found
class MineTeamFoundAP:
    def __init__(self, team_number_found ):
        self.team_number_found = team_number_found

    def __call__(self, action, obs, rew, done, info, cur_player):
        pos, *_ = obs

        caught = False
        for idx, position in enumerate(pos):
            if idx % 2 == self.team_number_found:
                if position != (-10, -10):
                    caught = True
        return caught


optimizers = {
    "Adam": Adam,
    "SGD": SGD
}

aut_funcs = {
    "full_fuel": full_fuel,
    "key": key,
    "goal": goal,
    "info_aut": MineInfoAutAP,
    "mine_inventory": MineInventoryAP,
    "mine_location": MineLocationAP,
    "mine_catch": MineCatchAP,
    "mine_catch_sides": MineCatchSidesAP,
    "mine_catch_sp": MineCatchSpecificAP,
    "mine_found_team": MineTeamFoundAP,
    "mine_found_agents": MineAllAgentsFoundAP,
    "mine_found_object": MineObjectFoundAP,
    "mine_found_by": MineThisAgentFoundBy,
    "mine_inv_ratio": MineInventoryRatioAP,

}

env_constructors = {
    "minecraft": MineWorldEnv.from_dict,
    "maze": FuelMazeEnv.from_dict,
    "gym": gym_make
}


def no_op_rewriter(x):
    return torch.Tensor([0.0])


training_nets = {
    "mazenet_v1": (mazenet_v1, maze_obs_rewrite_creator),
    "minenet_v1": (minenet_v1, mine_obs_rewriter_creator),
    "mine_mazenet_v1": (mine_mazenet_v1, mine_obs_rewriter_creator),
    "mine-mazenet-dirs": (mine_mazenet_dirs, mine_obs_rewriter_creator),
    "mine_mazenet_inv": (mine_mazenet_inv, mine_obs_rewriter_creator),
    "basicnet": (basic_net, lambda e: torch.Tensor),
    "no-op": (no_op_make, lambda e: no_op_rewriter)
}

curiosity_nets = {
    "mazernd_v1": (mazernd_v1, maze_obs_rewrite_creator),
    "minernd_v1": (minernd_v1, mine_obs_rewriter_creator),
    "no-op": (no_op_cur_make, no_op_rewriter)
}

loss_funcs = {
    "MCTS": TakeSimilarActionsLossFunction,
    "PPO": PPOLossFunction,
    "A2C": AdvantageActorCriticLossFunction
}

aut_transplant_anneals = {
    "Exponential": ExponentialAnnealedAutStats,
    "UCB": UCBAnnealedAutStats
}


def get_folders(checkpoint_base_path, log_path, run_name, postfix, teampostfixes):
    def interpolate2(text, run_name):
        if not text:
            return text

        if run_name and "%s" in text:
            return text % (run_name,)
        else:
            return text

    CHECKPOINT_PATHS = list(interpolate2(checkpoint_base_path, run_name) + postfix + teampostfix
                            for teampostfix in teampostfixes)

    LOG_FOLDERS = list(interpolate2(log_path, run_name) + postfix + teampostfix
                       for teampostfix in teampostfixes)

    return CHECKPOINT_PATHS, LOG_FOLDERS


def init_checkpoints(CHECKPOINT_PATHS, LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, DEVICE, TRANSPLANTS=None,
                     OLD_TRANSPLANTS=None, TRANSPLANT_FROM_LIST=None):
    try:

        cmans = list(CheckpointManager(CHECKPOINT_PATH, LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)
                     for CHECKPOINT_PATH in CHECKPOINT_PATHS)

    except EOFError:

        cmans = list(
            CheckpointManager(CHECKPOINT_PATH + "_copy", LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)
            for CHECKPOINT_PATH in CHECKPOINT_PATHS)

    if TRANSPLANTS is not None:
        for idx, cman in enumerate(cmans):  # TODO: check functionality for multi agent
            if TRANSPLANTS[idx]:
                cman = TransplantCheckpointManager(cman, TRANSPLANT_FROM_LIST[idx])
                cman.transplant(
                    "aut")  # Generating the automaton may not be completely deterministic, we want the same states
            elif OLD_TRANSPLANTS[idx]:
                cman = TransplantCheckpointManager(cman, TRANSPLANT_FROM_LIST[idx])
                for field in OLD_TRANSPLANTS[idx]:
                    cman.transplant(field)

    return cmans


def get_aut_config(config):
    def get_func(param: dict):
        func_or_generator = aut_funcs[param["func"]]
        func_params = param.get("params")
        if func_params is None:
            return func_or_generator
        else:
            return func_or_generator(**func_params)

    autconfig: List[dict] = config["automatons"]

    # list of specs, indexed by corresponding agent
    LTLF_SPECS = list(autconfig[i]["spec"] for i in range(len(autconfig)))

    # if we are doing the negation of the LTLF spec for
    # second player, we should be able to use loss functions
    AUT_PARAM_NAMES = list([param["name"] for param in aut["params"]] for aut in autconfig)

    AUT_PARAM_FUNCS = list([get_func(p) for p in aut["params"]] for aut in autconfig)

    AUT_STATS_PARAMS = list(aut.get("aut_stats_params", dict()) for aut in autconfig)

    AUT_OTHER_PARAMS = list({
                                "terminate_on_fail": aut.get("terminate_on_fail", True),
                                "termination_fail_reward": aut.get("termination_fail_reward", 0),
                                "terminate_on_accept": aut.get("terminate_on_accept", False),
                                "termination_accept_reward": aut.get("termination_accept_reward", 1),
                                "no_aut_shape": aut.get("no_aut_shape", False),
                            } for aut in autconfig)

    # Logging and checkpointing the postfix now for all agents
    teampostfixes = list(aut["checkpoint_postfix"] for aut in autconfig)

    return autconfig, LTLF_SPECS, AUT_PARAM_NAMES, AUT_PARAM_FUNCS, AUT_STATS_PARAMS, AUT_OTHER_PARAMS, teampostfixes


def init_auts(cmans, LTLF_SPECS, AUT_PARAM_NAMES, AUT_STATS_PARAMS):
    # should probably do this all the way around, just testing that it works for now,
    # listifying will follow after results
    auts = list(cman.load("aut", AutomatonSet.from_ltlf(LTLF_SPECS[i], AUT_PARAM_NAMES[i]), PickleLoadHandler())
                for i, cman in enumerate(cmans))

    aut_stats_list = list(cman.load("aut_stats", AutStats(len(auts[idx].graph.network), **AUT_STATS_PARAMS[idx]),
                                    StateDictLoadHandler())
                          for idx, cman in enumerate(cmans))

    return auts, aut_stats_list


def get_env_config(config, autconfig):
    envconfig = config["env"]
    MAX_EPISODE_LEN = envconfig["max_episode_len"]
    MAX_LEN_REWARD = envconfig.get("max_len_reward")
    ENV_CONFIG = envconfig["params"]
    ENV_TYPE = envconfig["type"]

    NUM_PLAYERS = ENV_CONFIG.get("num_agents", len(autconfig))
    NUM_TEAMS = ENV_CONFIG.get("num_teams", len(autconfig))

    AGENTS_BY_TEAM = list()
    for idx, aut in enumerate(autconfig):
        agent_default = list()
        agent_default.append(idx)
        AGENTS_BY_TEAM.append(aut.get("agents_following", agent_default))

    return MAX_EPISODE_LEN, MAX_LEN_REWARD, ENV_CONFIG, ENV_TYPE, NUM_PLAYERS, NUM_TEAMS, AGENTS_BY_TEAM


def init_env(auts, ENV_TYPE, ENV_CONFIG, AUT_PARAM_FUNCS, AUT_OTHER_PARAMS):
    # as many original environments as we have auts specified, one per team
    # need one env for team
    orig_envs = list(env_constructors[ENV_TYPE](ENV_CONFIG, i) for i in range(0, len(auts)))

    envs = list(AutShapingWrapperAdv(orig_envs[i], AUT_PARAM_FUNCS[i], aut,
                                     player_num=i, use_potential=False, **AUT_OTHER_PARAMS[i])
                for (i, aut) in enumerate(auts))

    return orig_envs, envs


def get_net_config(config):
    # Policy training hyperparameters
    training: dict = config["training"]

    LEARNING_RATE = training["learning_rate"]
    REPLAY_BUFFER = training["replay_buffer"]
    MIN_TRACE_TO_TRAIN = training["min_trace_to_train"]
    PPO_TRAIN_ROUNDS = training["train_rounds"]
    NETWORK = training.get("network", "mazenet_v1")
    NETWORK_PARAMS = training.get("params", dict())

    OPTIMIZER = optimizers[training.get("optimizer")]
    OPTIMIZER_PARAMS = training.get("opt_params", {})

    # Loss function
    loss: dict = config.get("loss")
    if loss:
        LOSS_FUNC = loss["type"]
        LOSS_PARAMS = loss.get("params", dict())
    else:
        LOSS_FUNC = "MCTS"
        LOSS_PARAMS = dict()

    if config.get("mcts"):
        config["episode_runner"] = {
            "type": "mcts_aut_episode",
            "params": config.pop("mcts")
        }

    return LEARNING_RATE, REPLAY_BUFFER, MIN_TRACE_TO_TRAIN, PPO_TRAIN_ROUNDS, NETWORK, NETWORK_PARAMS, OPTIMIZER, OPTIMIZER_PARAMS, LOSS_FUNC, LOSS_PARAMS


def get_episode_runner_config(config):
    # Policy runner parameters
    episode_runner_config = config["episode_runner"]
    EPISODE_RUNNER_TYPE = episode_runner_config["type"]
    EPISODE_RUNNER_PARAMS = episode_runner_config.get("params", dict())

    return EPISODE_RUNNER_TYPE, EPISODE_RUNNER_PARAMS


def init_net(cmans, orig_envs, envs, NETWORK, NETWORK_PARAMS, CURIOSITY_NET, CURIOSITY_PARAMS, DEVICE, NUM_PLAYERS,
             LOSS_FUNC, LOSS_PARAMS, LEARNING_RATE, OPTIMIZER, OPTIMIZER_PARAMS, DISCOUNT):
    train_net_creator, train_rewriter_creator = training_nets[NETWORK]

    nets = list(cman.load("net", train_net_creator(orig_envs[idx], **NETWORK_PARAMS),
                          CombinedLoadHandler(StateDictLoadHandler(), InitZeroLoadHandler())).to(DEVICE)
                for idx, cman in enumerate(cmans))

    for net in nets:
        net.share_memory()

    # not using this?? I think this is broken
    if CURIOSITY_NET:
        curiosity_net_creator, curiosity_rewriter_creator = curiosity_nets[CURIOSITY_NET]

        icm_list = list(
            cman.load("icm", curiosity_net_creator(orig_envs[idx], **CURIOSITY_PARAMS), StateDictLoadHandler()).to(
                DEVICE) for idx, cman in enumerate(cmans))
        for icm in icm_list:
            icm.share_memory()

        action_spaces = list(env.action_space.n for env in envs)
        icm_opt_list = list(cman.load("icm_opt",
                                      ModuleCuriosityOptimizer(icm_list[idx],
                                                               curiosity_rewriter_creator(orig_env),
                                                               action_spaces[idx], CURIOSITY_LEARNING_RATE,
                                                               DEVICE), StateDictLoadHandler())
                            for idx, cman in enumerate(cmans))
    else:
        icm = None
        icm_opt_list = list(NoopCuriosityOptimizer() for i in range(0, NUM_PLAYERS))

    loss_functions = list(loss_funcs[LOSS_FUNC](net=nets[i], device=DEVICE, discount=DISCOUNT, **LOSS_PARAMS)
                          for i in range(0, len(nets)))

    optimizers = list(cman.load("opt", OPTIMIZER(nets[idx].parameters(), lr=LEARNING_RATE, **OPTIMIZER_PARAMS),
                                StateDictLoadHandler()) for idx, cman in enumerate(cmans))

    train_state_rewriter = train_rewriter_creator(orig_envs[0])

    return nets, train_state_rewriter, icm_opt_list, icm, loss_functions, optimizers


if __name__ == '__main__':
    import argparse
    import json5 as json

    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--log")
    p.add_argument("--checkpoint")
    p.add_argument("--run-name")
    p.add_argument("--do-not-load-from-checkpoint", dest="load_checkpoint", action="store_false")
    p.add_argument("--do-not-save-checkpoint", dest="save_checkpoint", action="store_false")
    p.add_argument("--checkpoint-every", default=1)
    p.add_argument("--workers", default=8)
    p.add_argument("--post", help="Add a postfix to the checkpoint and tensorboard names")
    p.add_argument("--stop-after", dest="stop_after",
                   help="Stop after roughly a certain number of steps have been reached")
    p.add_argument("--hide-play-step", dest="hide_play_step", action='store_true',
                   help="Do not render each play step")

    args = vars(p.parse_args())

    run_name = args.get("run_name")
    STOP_AFTER = args.get("stop_after")
    if STOP_AFTER:
        STOP_AFTER = int(STOP_AFTER)


    def interpolate(text):
        if not text:
            return text

        if run_name and "%s" in text:
            return text % (run_name,)
        else:
            return text


    config_file = interpolate(args["config"])

    postfix = ""

    if args.get("post"):
        postfix = "_" + args["post"]

    with open(config_file) as f:
        config = json.load(f)

    autconfig, LTLF_SPECS, AUT_PARAM_NAMES, AUT_PARAM_FUNCS, AUT_STATS_PARAMS, AUT_OTHER_PARAMS, teampostfixes = get_aut_config(
        config)

    DISCOUNT = config["discount"]

    if "maze" in config:
        maze = config["maze"]

        config["env"] = dict()
        config["env"]["type"] = "maze"
        config["env"]["max_episode_len"] = maze["max_episode_len"]
        del maze["max_episode_len"]
        config["env"]["params"] = maze
        del config["maze"]

    MAX_EPISODE_LEN, MAX_LEN_REWARD, ENV_CONFIG, \
    ENV_TYPE, NUM_PLAYERS, NUM_TEAMS, AGENTS_BY_TEAM = get_env_config(config, autconfig)

    LEARNING_RATE, REPLAY_BUFFER, MIN_TRACE_TO_TRAIN, PPO_TRAIN_ROUNDS, \
    NETWORK, NETWORK_PARAMS, OPTIMIZER, OPTIMIZER_PARAMS, LOSS_FUNC, LOSS_PARAMS = get_net_config(config)

    # Policy runner parameters
    EPISODE_RUNNER_TYPE, EPISODE_RUNNER_PARAMS = get_episode_runner_config(config)
    EPISODE_RUNNER_PARAMS["show_play_step"] = not args["hide_play_step"]

    # Curiosity Parameters
    curiosity: dict = config.get("curiosity")

    if curiosity:
        if "feature_space" in curiosity:
            curiosity["type"] = "mazernd_v1"
            curiosity["params"] = {"feature_space": curiosity["feature_space"]}
            del curiosity["feature_space"]

        CURIOSITY_LEARNING_RATE = curiosity["learning_rate"]
        CURIOSITY_NET = curiosity["type"]
        CURIOSITY_PARAMS = curiosity.get("params", dict())
    else:
        CURIOSITY_NET = None
        CURIOSITY_PARAMS = None

    CHECKPOINT_EVERY = int(args["checkpoint_every"])

    checkpoint_base_path = args.get("checkpoint")
    log_path = args.get("log")
    CHECKPOINT_PATHS, LOG_FOLDERS = get_folders(checkpoint_base_path, log_path, run_name, postfix, teampostfixes)

    """
    There are two types of "transplants":
    1. "Old" transplant, this just literally loads the state from the "from" checkpoint instead of creating the state
    from scratch
    2. "Regular" transplant, this is only for the automaton statistics, and it anneals between the imported values and
    the values created during this run."""

    # this is a list now, possibly multiple transplants
    # TODO: check this works
    transplant_configs = config.get("transplant")
    TRANSPLANTS = [False] * NUM_PLAYERS
    OLD_TRANSPLANTS: List[Union[bool, List[str]]] = [False] * NUM_PLAYERS
    TRANSPLANT_FROM_LIST = [None] * NUM_PLAYERS
    ANNEAL_AUT_TRANSPLANTS = [None] * NUM_PLAYERS
    ANNEAL_AUT_TRANSPLANTS_PARAMS = [None] * NUM_PLAYERS

    if transplant_configs:
        for idx, transplant_config in enumerate(transplant_configs):
            # TODO: configure transplant_config for multi agent
            if transplant_config:
                TRANSPLANT_FROM_LIST[idx] = transplant_config["from"]
                if transplant_config.get("fields"):
                    OLD_TRANSPLANTS[idx] = transplant_config["fields"]
                else:
                    TRANSPLANTS[idx] = True
                    aut_transplant = transplant_config["automaton"]
                    ANNEAL_AUT_TRANSPLANTS[idx] = aut_transplant["type"]
                    ANNEAL_AUT_TRANSPLANTS_PARAMS[idx] = aut_transplant.get("params", {})

    for idx, CHECKPOINT_PATH in enumerate(CHECKPOINT_PATHS):
        # assuming we run bots checkpointed at the same place, packaged
        if CHECKPOINT_PATH:

            LOAD_FROM_CHECKPOINT = args["load_checkpoint"]

            if not os.path.isfile(CHECKPOINT_PATH):
                LOAD_FROM_CHECKPOINT = False
                print("NOTE: no existing checkpoint found for agent " + str(idx) +
                      ", will create new one if checkpoint saving is enabled.")
            else:
                if OLD_TRANSPLANTS[idx]:
                    OLD_TRANSPLANTS[idx] = False
                    print("NOTE: Loading agent " + str(idx) +
                          "from checkpoint, so transplant disabled")

            SAVE_CHECKPOINTS = args["save_checkpoint"]
        else:
            CHECKPOINT_PATHS = [None] * NUM_TEAMS  # TODO: might have to change this
            LOAD_FROM_CHECKPOINT = False
            SAVE_CHECKPOINTS = False
            if not args["save_checkpoint"]:
                print("WARNING: This run is not being checkpointed! Use --do-not-save-checkpoint to suppress.")

    NUM_PROCESSES = int(args["workers"])
    DEVICE = torch.device(args["device"])


def run_mcts_aut_episode(nets: List[torch.nn.Module], envs: List[AutShapingWrapperAdv], max_length: int,
                         max_len_reward: Union[int, None],
                         curiosity: ModuleCuriosityOptimizer,
                         device, c_puct, c_aut,
                         num_batches, batch_size, stats: List[AutStats],
                         train_state_rewriter: Callable[[Any], torch.Tensor],
                         state_observer: Callable[[Any], None] = None, c_sigma=1, c_intrins=1, render_every_frame=True,
                         **kwargs) \
        -> Tuple[List[TraceStep], float]:
    """
    Run an episode using MCTS with curiosity as the action selection
    :param net: The policy/value network
    :param env: The environment to run the simulation in
    :param max_length: When to cut off the simulation
    :param curiosity: Something to calculate the relative "newness" of a state
    :param device: The device to run the simulation on
    :param c_puct: Puct constant of MCTS
    :param num_batches: How many groups of MCTS sims to run
    :param batch_size: How many MCTS sims per group
    :param state_observer: Function to call for every state seen
    :return: A trace and final value estimate
    """

    def curiosity_evaluator(sars):
        states, actions, rewards, next_states, _ = zip(*sars)
        rewards = curiosity.get_curiosity(states, actions, next_states)
        return rewards.tolist()

    def curiosity_trainer(sars):
        states, actions, rewards, next_states, _ = zip(*sars)
        curiosity.train(states, actions, next_states, train_rounds=1)

    def state_evaluator(states, agent):
        # agent = (agent+1)%2
        states_transformed = torch.stack(tuple(train_state_rewriter(s) for s in states))
        pols, vals = nets[agent](states_transformed.to(device))
        pollist = F.softmax(pols, dim=-1).tolist()
        vallist = vals.squeeze(-1).tolist()

        return list(zip(pollist, vallist))

    for stat in stats:
        stat.synchronize()

    mcts = MCTSAutAdvList(envs[0].action_space.n, curiosity_evaluator, state_evaluator, curiosity_trainer,
                          c_puct=c_puct,
                          aut_stats=stats, c_aut=c_aut, c_sigma=c_sigma, c_intrins=c_intrins, **kwargs)

    def action_value_generator(state, step):
        mcts.mcts_batch(envs, state, num_batches, batch_size)
        probs, values = mcts.get_policy_value(state, 1)
        return probs, list(max(value) for value in values)

    return run_episode_generic_adv(envs, action_value_generator, max_length, max_len_reward,
                                   ptan.actions.ProbabilityActionSelector(),
                                   state_observer)


def run_mcts_aut_team_episode(nets: List[torch.nn.Module], envs: List[AutShapingWrapperAdv], max_length: int,
                              max_len_reward: Union[int, None],
                              curiosity: ModuleCuriosityOptimizer,
                              device, c_puct, c_aut,
                              num_batches, batch_size, stats: List[AutStats],
                              train_state_rewriter: Callable[[Any], torch.Tensor],
                              state_observer: Callable[[Any], None] = None, c_sigma=1, c_intrins=1,
                              render_every_frame=True,
                              show_play_step=True,
                              **kwargs) \
        -> Tuple[List[TraceStep], float]:
    """
    Run an episode using MCTS with curiosity as the action selection
    :param net: The policy/value network
    :param env: The environment to run the simulation in
    :param max_length: When to cut off the simulation
    :param curiosity: Something to calculate the relative "newness" of a state
    :param device: The device to run the simulation on
    :param c_puct: Puct constant of MCTS
    :param num_batches: How many groups of MCTS sims to run
    :param batch_size: How many MCTS sims per group
    :param state_observer: Function to call for every state seen
    :return: A trace and final value estimate
    """

    def curiosity_evaluator(sars):
        states, actions, rewards, next_states, _ = zip(*sars)
        rewards = curiosity.get_curiosity(states, actions, next_states)
        return rewards.tolist()

    def curiosity_trainer(sars):
        states, actions, rewards, next_states, _ = zip(*sars)
        curiosity.train(states, actions, next_states, train_rounds=1)

    def state_evaluator(states, team):
        # agent = (agent+1)%2
        states_transformed = torch.stack(tuple(train_state_rewriter(s) for s in states))
        pols, vals = nets[team](states_transformed.to(device))
        pollist = F.softmax(pols, dim=-1).tolist()
        vallist = vals.squeeze(-1).tolist()

        return list(zip(pollist, vallist))


    for stat in stats:
        stat.synchronize()

    dont_use_aut = list(e.no_aut_shape for e in envs)
    team_membership = envs[0].env.teams
    mcts = MCTSAutAdvList(envs[0].action_space.n, curiosity_evaluator, state_evaluator, curiosity_trainer,
                          c_puct=c_puct,
                          aut_stats=stats, c_aut=c_aut, c_sigma=c_sigma, c_intrins=c_intrins,
                          dont_use_aut=dont_use_aut,
                          team_membership=team_membership,
                          **kwargs)

    def action_value_generator(state, step):
        mcts.mcts_batch(envs, state, num_batches, batch_size)
        probs, values = mcts.get_policy_value(state, 1)

        return probs, list(max(value) for value in values)

    return run_episode_generic_team(envs, action_value_generator, max_length, max_len_reward,
                                    ptan.actions.ProbabilityActionSelector(),
                                    state_observer, show_play_step=show_play_step)


def run_exec_team_episode(nets: List[torch.nn.Module], envs: List[AutShapingWrapperAdv], max_length: int,
                              max_len_reward: Union[int, None],
                              curiosity: ModuleCuriosityOptimizer,
                              device, c_puct, c_aut,
                              num_batches, batch_size, stats: List[AutStats],
                              train_state_rewriter: Callable[[Any], torch.Tensor],
                              state_observer: Callable[[Any], None] = None, c_sigma=1, c_intrins=1,
                              render_every_frame=True,
                              show_play_step=True,
                              **kwargs) \
        -> Tuple[List[TraceStep], float]:
    """
    Run an episode using MCTS with curiosity as the action selection
    :param net: The policy/value network
    :param env: The environment to run the simulation in
    :param max_length: When to cut off the simulation
    :param curiosity: Something to calculate the relative "newness" of a state
    :param device: The device to run the simulation on
    :param c_puct: Puct constant of MCTS
    :param num_batches: How many groups of MCTS sims to run
    :param batch_size: How many MCTS sims per group
    :param state_observer: Function to call for every state seen
    :return: A trace and final value estimate
    """


    # states should be a list containing a single state.
    def get_pi_cnn(state, team):

        states_transformed = torch.stack(tuple([train_state_rewriter(
            state)]))  # We don't need the stack, but will keep it since it makes the dimensions right

        net = nets[team]

        (rawprobs, rawvals) = net(states_transformed.to(device))
        probs = F.softmax(rawprobs, dim=-1).tolist()
        vals = rawvals.squeeze(-1).tolist()

        return probs[0]

    for stat in stats:
        stat.synchronize()


    return run_episode_generic_team_cnn_only(envs, get_pi_cnn, max_length, max_len_reward,
                                    ptan.actions.ArgmaxActionSelector(),
                                    state_observer, show_play_step=show_play_step)




# TODO: Change run_mcts_aut_team_episode to call this function to avoid duplication (getting the parameters right will be tedious)
def get_mcts_aut_team(nets: List[torch.nn.Module], envs: List[AutShapingWrapperAdv], max_length: int,
                      max_len_reward: Union[int, None],
                      curiosity: ModuleCuriosityOptimizer,
                      device, c_puct, c_aut,
                      num_batches, batch_size, stats: List[AutStats],
                      train_state_rewriter: Callable[[Any], torch.Tensor],
                      state_observer: Callable[[Any], None] = None, c_sigma=1, c_intrins=1, render_every_frame=True,
                      show_play_step=True,
                      **kwargs):
    """
    Run an episode using MCTS with curiosity as the action selection
    :param net: The policy/value network
    :param env: The environment to run the simulation in
    :param max_length: When to cut off the simulation
    :param curiosity: Something to calculate the relative "newness" of a state
    :param device: The device to run the simulation on
    :param c_puct: Puct constant of MCTS
    :param num_batches: How many groups of MCTS sims to run
    :param batch_size: How many MCTS sims per group
    :param state_observer: Function to call for every state seen
    :return: A trace and final value estimate
    """

    def curiosity_evaluator(sars):
        states, actions, rewards, next_states, _ = zip(*sars)
        rewards = curiosity.get_curiosity(states, actions, next_states)
        return rewards.tolist()

    def curiosity_trainer(sars):
        states, actions, rewards, next_states, _ = zip(*sars)
        curiosity.train(states, actions, next_states, train_rounds=1)

    def state_evaluator(states, team):
        # agent = (agent+1)%2
        states_transformed = torch.stack(tuple(train_state_rewriter(s) for s in states))
        pols, vals = nets[team](states_transformed.to(device))
        pollist = F.softmax(pols, dim=-1).tolist()
        vallist = vals.squeeze(-1).tolist()

        return list(zip(pollist, vallist))

    for stat in stats:
        stat.synchronize()

    dont_use_aut = list(e.no_aut_shape for e in envs)
    team_membership = envs[0].env.teams
    mcts = MCTSAutAdvList(envs[0].action_space.n, curiosity_evaluator, state_evaluator, curiosity_trainer,
                          c_puct=c_puct,
                          team_membership = team_membership,
                          aut_stats=stats, c_aut=c_aut, c_sigma=c_sigma, c_intrins=c_intrins,
                            **kwargs)
    return mcts


# Code would have to be edited for this to work.
def run_aut_episode(nets: List[torch.nn.Module], envs: List[AutShapingWrapperAdv], max_length: int,
                    max_len_reward: Optional[int],
                    curiosity: List[ModuleCuriosityOptimizer], device,
                    train_state_rewriter: Callable[[Any], torch.Tensor], stats: List[AutStats],
                    state_observer: Callable[[Any], None] = None, render_every_frame=True) -> Tuple[
    List[TraceStep], float]:
    for stat in stats:
        stat.synchronize()

    def action_value_generator(state, step):
        obs_tensor = train_state_rewriter(state).to(device)
        obs_batch = obs_tensor.unsqueeze(dim=0)
        probs, values = nets[state[3]](obs_batch)
        pols_soft = F.softmax(probs.double(), dim=-1).squeeze(0)
        pols_soft /= pols_soft.sum()
        pols_soft = pols_soft.tolist()
        val = values.squeeze(0).tolist()[0]
        if render_every_frame:
            env.render()

        return pols_soft, val

    # TODO curiosity and automaton bonuses
    return run_episode_generic(env, action_value_generator, max_length, max_len_reward,
                               ptan.actions.EpsilonGreedyActionSelector(
                                   selector=ptan.actions.ProbabilityActionSelector(),
                                   epsilon=.1),
                               state_observer)


episode_runners = {
    "aut_episode": run_aut_episode,
    "mcts_adv": run_mcts_aut_episode,
    "mcts_team": run_mcts_aut_team_episode,
    "exec_no_mcts": run_exec_team_episode,
}


def get_team(player):
    return player % 2


def get_env_state(envs, partial=None):
    player = envs[0].get_turnn()
    team = envs[0].get_team_turn()
    state = envs[team].get_obs(partial)

    return player, team, state


def do_env_step(envs, action):
    team = envs[0].get_team_turn()

    # Set the return value from the current team's environment
    for i, env in enumerate(envs):
        if i == team:
            retval = env.step(action)
        else:
            env.step(action)

    return retval


def get_player_env(envs, player):
    return envs[get_team(player)]


def get_player_net(nets, player):
    return nets[get_team(player)]


def init_game_envsonly(config_file):
    import json5 as json

    with open(config_file) as f:
        config = json.load(f)

    autconfig, LTLF_SPECS, AUT_PARAM_NAMES, AUT_PARAM_FUNCS, AUT_STATS_PARAMS, AUT_OTHER_PARAMS, teampostfixes = get_aut_config(
        config)
    MAX_EPISODE_LEN, MAX_LEN_REWARD, ENV_CONFIG, ENV_TYPE, NUM_PLAYERS, NUM_TEAMS, AGENTS_BY_TEAM = get_env_config(
        config,
        autconfig)

    auts = [AutomatonSet.from_ltlf(LTLF_SPECS[i], AUT_PARAM_NAMES[i]) for i in range(len(LTLF_SPECS))]
    orig_envs, envs = init_env(auts, ENV_TYPE, ENV_CONFIG, AUT_PARAM_FUNCS, AUT_OTHER_PARAMS)

    for env in envs:
        env.reset()

    return envs


# partial=None means use default config, True/False means Partial/Full
def init_game(config_file, checkpoint_base_path, run_name, postfix, device, partial=None):
    import json5 as json

    with open(config_file) as f:
        config = json.load(f)

    autconfig, LTLF_SPECS, AUT_PARAM_NAMES, AUT_PARAM_FUNCS, AUT_STATS_PARAMS, AUT_OTHER_PARAMS, teampostfixes = get_aut_config(config)

    MAX_EPISODE_LEN, MAX_LEN_REWARD, ENV_CONFIG, ENV_TYPE, NUM_PLAYERS, NUM_TEAMS, AGENTS_BY_TEAM = get_env_config(config, autconfig)

    EPISODE_RUNNER_TYPE, EPISODE_RUNNER_PARAMS = get_episode_runner_config(config)

    LEARNING_RATE, REPLAY_BUFFER, MIN_TRACE_TO_TRAIN, PPO_TRAIN_ROUNDS, NETWORK, \
    NETWORK_PARAMS, OPTIMIZER, OPTIMIZER_PARAMS, LOSS_FUNC, LOSS_PARAMS = get_net_config(config)
    DISCOUNT = config["discount"]

    num_batches = EPISODE_RUNNER_PARAMS["num_batches"]
    batch_size = EPISODE_RUNNER_PARAMS["batch_size"]

    if checkpoint_base_path:
        CHECKPOINT_PATHS, _ = get_folders(checkpoint_base_path, "", run_name, postfix, teampostfixes)
        LoadCheckpoint = True
    else:
        CHECKPOINT_PATHS = [None] * NUM_TEAMS
        LoadCheckpoint = False
    cmans = init_checkpoints(CHECKPOINT_PATHS, LoadCheckpoint, False, "cpu")
    auts, aut_stats_list = init_auts(cmans, LTLF_SPECS, AUT_PARAM_NAMES, AUT_STATS_PARAMS)
    orig_envs, envs = init_env(auts, ENV_TYPE, ENV_CONFIG, AUT_PARAM_FUNCS, AUT_OTHER_PARAMS)
    nets, train_state_rewriter, icm_opt_list, icm, loss_functions, optimizers = init_net(cmans, orig_envs, envs,
                                                                                         NETWORK, NETWORK_PARAMS, None,
                                                                                         None, device, NUM_PLAYERS,
                                                                                         LOSS_FUNC, LOSS_PARAMS,
                                                                                         LEARNING_RATE, OPTIMIZER,
                                                                                         OPTIMIZER_PARAMS, DISCOUNT)

    # wrapped_aut_stats is only different if there are TRANSPLANTs - ignore this for now
    wrapped_aut_stats = aut_stats_list

    # We assume there is no transplanting, and so use the unwrapped aut_stats
    icm_opt = NoopCuriosityOptimizer()

    tau = 1

    for env in envs:
        env.reset()

    dont_use_aut = list(e.no_aut_shape for e in envs)

    # This should match the parameters passed to get_parallel_queue_adv in run()
    mcts = \
        get_mcts_aut_team(nets=nets, envs=envs, max_length=MAX_EPISODE_LEN, max_len_reward=MAX_LEN_REWARD,
                          curiosity=icm_opt, state_observer=None, device=device,
                          stats=wrapped_aut_stats, train_state_rewriter=train_state_rewriter,
                          dont_use_aut=dont_use_aut,
                          **EPISODE_RUNNER_PARAMS)

    # Define various functions to return to caller.
    # The functions will take envs as a parameter to give flexibility. But, we could just use the envs defined here

    def get_pi_cnn(envs):
        player, team, state = get_env_state(envs, partial)

        states_transformed = torch.stack(tuple([train_state_rewriter(
            state)]))  # We don't need the stack, but will keep it since it makes the dimensions right

        net = nets[team]

        (rawprobs, rawvals) = net(states_transformed.to(device))
        probs = F.softmax(rawprobs, dim=-1).tolist()
        vals = rawvals.squeeze(-1).tolist()

        return probs[0], vals[0]

    # def get_mcts():
    #     # This should match the parameters passed to get_parallel_queue_adv in run()
    #     mcts, action_value_generator = \
    #           get_mcts_aut_team( nets=nets, envs=envs, max_length=MAX_EPISODE_LEN, max_len_reward=MAX_LEN_REWARD,
    #                              curiosity=icm_opt, state_observer=None, device=device,
    #                              stats=wrapped_aut_stats, train_state_rewriter=train_state_rewriter,
    #                              **EPISODE_RUNNER_PARAMS)
    #
    #     return mcts

    def do_mcts_batch(envs):
        # We will assume that mcts_batch will save/restore the environment state - maybe we should assert the envs are the same when mcts returns
        player, team, state = get_env_state(envs, partial)
        mcts.mcts_batch(envs, state, num_batches, batch_size)

    def get_tree_and_play_policies(envs):
        player, team, state = get_env_state(envs, partial)

        pi_tree, pi_tree_score, Q, Y, pi_cnn = mcts.pick_action(state, True)

        pi_play, _ = mcts.get_policy_value(state,
                                           tau)  # The Q returned by this function should be the same as pick_Action

        return pi_tree, pi_tree_score, Q, Y, pi_play, pi_cnn

    return envs, get_pi_cnn, do_mcts_batch, get_tree_and_play_policies


def run():
    torch.multiprocessing.set_start_method("spawn", force=True)
    signal.signal(signal.SIGHUP, throwKeyInterr)

    cmans = init_checkpoints(CHECKPOINT_PATHS, LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, DEVICE, TRANSPLANTS,
                             OLD_TRANSPLANTS, TRANSPLANT_FROM_LIST)

    auts, aut_stats_list = init_auts(cmans, LTLF_SPECS, AUT_PARAM_NAMES, AUT_STATS_PARAMS)
    orig_envs, envs = init_env(auts, ENV_TYPE, ENV_CONFIG, AUT_PARAM_FUNCS, AUT_OTHER_PARAMS)
    nets, train_state_rewriter, icm_opt_list, icm, loss_functions, optimizers = init_net(cmans, orig_envs, envs,
                                                                                         NETWORK, NETWORK_PARAMS,
                                                                                         CURIOSITY_NET,
                                                                                         CURIOSITY_PARAMS, DEVICE,
                                                                                         NUM_PLAYERS, LOSS_FUNC,
                                                                                         LOSS_PARAMS, LEARNING_RATE,
                                                                                         OPTIMIZER, OPTIMIZER_PARAMS,
                                                                                         DISCOUNT)

    # quick fix for trivial acceptance states for adversarial automata
    # envs[1].terminate_on_accept = False

    #        in a for loop (many agent approach)

    writers = list(SummaryWriter(LOG_FOLDER) for LOG_FOLDER in LOG_FOLDERS)

    # TODO: should be the same train loop for all checkpoints, all writers included
    train_loops = list(cman.load("train_loop",
                                 RandomReplayTrainingLoop(DISCOUNT, REPLAY_BUFFER, MIN_TRACE_TO_TRAIN, PPO_TRAIN_ROUNDS,
                                                          train_state_rewriter, writers, DEVICE),
                                 StateDictLoadHandler()) for idx, cman in enumerate(cmans))

    elos = list(train_loops[team].elo_ratings[team] for team in range(0, len(envs)))

    train_loop = train_loops[0]
    train_loop.elo_ratings[1] = elos[1]


    wrapped_aut_stats = [None] * NUM_TEAMS

    for idx, cman in enumerate(cmans):
        # TODO:  transplant multi agent
        if TRANSPLANTS[idx]:
            orig_alt_stats = cman.load_from_alt("aut_stats", AutStats(len(auts[idx].graph.network)),
                                                StateDictLoadHandler())
            wrapped_aut_stats[idx] = aut_transplant_anneals[ANNEAL_AUT_TRANSPLANTS[idx]](orig_alt_stats,
                                                                                         aut_stats_list[idx],
                                                                                         **
                                                                                         ANNEAL_AUT_TRANSPLANTS_PARAMS[
                                                                                             idx])
            wrapped_aut_stats[idx].set_step(train_loop.num_rounds)
            train_loop.add_round_hook(wrapped_aut_stats[idx].set_step)
        else:
            wrapped_aut_stats[idx] = aut_stats_list[idx]

    # same function for all players, just specify playerNum when called
    def aut_hook(trace: List[TraceReturnStep], final_value, team_number: int):
        # Only count each state once per run
        prev_edges = set()
        last_state = None
        # this was commented TODO when i initially received code: bb912...does it?
        for trst in trace:
            # for trst in reversed(trace):  # TODO does this need to be reversed?
            this_state = frozenset(trst.info["automaton_states"])

            if len(this_state) > 0:
                this_state = set(this_state).pop()
            else:
                this_state = None

            edge = (last_state, this_state)
            last_state = this_state
            if edge[0] is not None and edge[1] is not None:
                if edge not in prev_edges:
                    aut_stats_list[team_number].visit(edge, trst.discounted_return)
                    prev_edges.add(edge)

    # use only 1 train_loop, specify aut hooks for each agent with team_num param
    train_loop.add_trace_hook(aut_hook)

    with get_parallel_queue_adv(num_processes=NUM_PROCESSES, episode_runner=episode_runners[EPISODE_RUNNER_TYPE],
                                nets=nets, envs=envs, max_length=MAX_EPISODE_LEN, max_len_reward=MAX_LEN_REWARD,
                                curiosity=icm_opt_list, state_observer=None, device=DEVICE,
                                stats=wrapped_aut_stats, train_state_rewriter=train_state_rewriter,
                                **EPISODE_RUNNER_PARAMS) as sim_round_queue:
        # random.seed(798)

        while True:

            train_loop(sim_round_queue, loss_functions, optimizers)

            if train_loop.num_rounds % CHECKPOINT_EVERY == 0:
                # print("num_rounds=", train_loop.num_rounds)
                list_dicts = []

                # ONE checkpoint per team
                for idx in range(0, NUM_TEAMS):
                    list_dicts.append({
                        "net": nets[idx],
                        "opt": optimizers[idx],
                        "train_loop": train_loop,
                        "aut_stats": aut_stats_list[idx],
                        "aut": auts[idx],
                    })

                if CURIOSITY_NET:
                    for idx, save_dict in enumerate(list_dicts):
                        save_dict.update({
                            "icm": icm_list[idx],
                            "icm_opt": icm_opt_list[idx]
                        })

                for idx, save_dict in enumerate(list_dicts):
                    cmans[idx].save(save_dict)

                if STOP_AFTER and train_loop.global_step > STOP_AFTER:
                    print("STOPPING: step limit " + str(train_loop.global_step) + "/" + str(STOP_AFTER))
                    break


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run()
