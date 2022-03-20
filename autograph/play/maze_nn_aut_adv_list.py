import math
import os
import signal
import sys
from typing import Callable, Any, Tuple, List, Union, Optional

import ptan
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import multiprocessing
from torch.optim import Adam, SGD

import autograph.lib.envs.mazeenv
from autograph.lib.automata import AutomatonSet
from autograph.lib.envs.mazeenv import FuelMazeEnv, FuelMazeObservation
from autograph.lib.envs.mazeenv import transform_coordinate
from autograph.lib.envs.mineworldenv_adv_list import MineWorldEnv
from autograph.lib.loss_functions import TakeSimilarActionsLossFunction, PPOLossFunction, \
    AdvantageActorCriticLossFunction
from autograph.lib.mcts_aut_adv_list import MCTSAutAdvList, AutStats, ExponentialAnnealedAutStats, UCBAnnealedAutStats
from autograph.lib.running import get_parallel_queue, get_parallel_queue_adv, RandomReplayTrainingLoop, run_episode_generic_adv, run_episode_generic
from autograph.lib.shaping import AutShapingWrapperAdv
from autograph.lib.util import element_add
from autograph.lib.util.checkpoint_manager import CheckpointManager, StateDictLoadHandler, CombinedLoadHandler, \
    InitZeroLoadHandler, PickleLoadHandler, TransplantCheckpointManager
from autograph.lib.util.trace_return_step import TraceStep, TraceReturnStep
from autograph.net.curiosity.curiosity_optimizer import ModuleCuriosityOptimizer, NoopCuriosityOptimizer
from autograph.net.maze_constructors import mazenet_v1, mazernd_v1, maze_obs_rewrite_creator
from autograph.net.mine_constructors import minenet_v1, mine_obs_rewriter_creator, minernd_v1, mine_mazenet_v1, mine_mazenet_inv
from autograph.net.misc_constructors import gym_make, no_op_cur_make, basic_net, no_op_make
import random

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
    def __init__(self, apname: str = None, ap_name: str = None):
        if not (apname or ap_name):
            raise ValueError("Did not provide ap_name to info aut")
        self.name = apname or ap_name

    def __call__(self, action, obs, rew, done, info, cur_player):
        return self.name in info["atomic_propositions"]


class MineInventoryAP:
    def __init__(self, inventory_item, quantity, player):
        self.item = inventory_item
        self.quantity = quantity
        self.player = player

    def __call__(self, action, obs, rew, done, info, cur_player):
        return cur_player == self.player and \
               info["inventory"][self.item] == self.quantity


class MineLocationAP:
    def __init__(self, location, player):
        self.location = tuple(location)
        self.player = player

    def __call__(self, action, obs, rew, done, info, cur_player):
        position, *_ = obs
        return position[self.player] == self.location


'''Two agents on same tile'''
class MineCatchAP:
    def __init__(self, players):
        # could be used in future to determine which players are on one tile
        self.players = tuple(players)

    def __call__(self, action, obs, rew, done, info, cur_player):
        position, *_ = obs
        return position[obs[3]] == position[(obs[3]+1) % 2]

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
    "mine_catch": MineCatchAP
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

    aut: dict = config["automaton"]

    # list of specs, indexed by corresponding agent
    LTLF_SPECS = aut["specs"]


    # if we are doing the negation of the LTLF spec for
    # second player, we should be able to use loss functions
    AUT_PARAM_NAMES = [param["name"] for param in aut["params"]]


    def get_func(param: dict):
        func_or_generator = aut_funcs[param["func"]]
        func_params = param.get("params")
        if func_params is None:
            return func_or_generator
        else:
            return func_or_generator(**func_params)


    AUT_PARAM_FUNCS = [get_func(p) for p in aut["params"]]

    AUT_OTHER_PARAMS = {
        "terminate_on_fail": aut.get("terminate_on_fail", True),
        "termination_fail_reward": aut.get("termination_fail_reward", 0),
        "terminate_on_accept": aut.get("terminate_on_accept", False),
        "termination_accept_reward": aut.get("termination_accept_reward", 1)
    }

    AUT_STATS_PARAMS = aut.get("aut_stats_params", dict())

    DISCOUNT = config["discount"]

    if "maze" in config:
        maze = config["maze"]

        config["env"] = dict()
        config["env"]["type"] = "maze"
        config["env"]["max_episode_len"] = maze["max_episode_len"]
        del maze["max_episode_len"]
        config["env"]["params"] = maze
        del config["maze"]

    env = config["env"]
    MAX_EPISODE_LEN = env["max_episode_len"]
    MAX_LEN_REWARD = env.get("max_len_reward")
    ENV_CONFIG = env["params"]
    ENV_TYPE = env["type"]

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

    # Policy runner parameters
    episode_runner = config["episode_runner"]
    EPISODE_RUNNER_TYPE = episode_runner["type"]
    EPISODE_RUNNER_PARAMS = episode_runner.get("params", dict())

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

    # Logging and checkpointing the postfix now only applies to

    postfixAgent0 = "_a0"
    postfixAgent1 = "_a1"

    LOG_FOLDER = interpolate(args.get("log")) + postfix + postfixAgent0
    LOG1_FOLDER = interpolate(args.get("log")) + postfix + postfixAgent1


    CHECKPOINT_EVERY = int(args["checkpoint_every"])

    CHECKPOINT_PATH = interpolate(args.get("checkpoint")) + postfix + postfixAgent0
    CHECKPOINT_PATH_1 = interpolate(args.get("checkpoint")) + postfix + postfixAgent1

    """
    There are two types of "transplants":
    1. "Old" transplant, this just literally loads the state from the "from" checkpoint instead of creating the state
    from scratch
    2. "Regular" transplant, this is only for the automaton statistics, and it anneals between the imported values and
    the values created during this run."""
    transplant_config = config.get("transplant")
    TRANSPLANT = False
    OLD_TRANSPLANT: Union[bool, List[str]] = False


    #TODO: configure transplant_config for multi agent
    if transplant_config:
        TRANSPLANT_FROM = transplant_config["from"]
        if transplant_config.get("fields"):
            OLD_TRANSPLANT = transplant_config["fields"]
        else:
            TRANSPLANT = True
            aut_transplant = transplant_config["automaton"]
            ANNEAL_AUT_TRANSPLANT = aut_transplant["type"]
            ANNEAL_AUT_TRANSPLANT_PARAMS = aut_transplant.get("params", {})


    #assuming we run bots checkpointed at the same place, packaged
    if CHECKPOINT_PATH:

        LOAD_FROM_CHECKPOINT = args["load_checkpoint"]
        if not os.path.isfile(CHECKPOINT_PATH):
            LOAD_FROM_CHECKPOINT = False
            print("NOTE: no existing checkpoint found, will create new one if checkpoint saving is enabled.")
        else:
            if OLD_TRANSPLANT:
                OLD_TRANSPLANT = False
                print("NOTE: Loading from checkpoint, so transplant disabled")

        SAVE_CHECKPOINTS = args["save_checkpoint"]
    else:
        CHECKPOINT_PATH = None
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
                         num_batches, batch_size, stats: List[AutStats], train_state_rewriter: Callable[[Any], torch.Tensor],
                         state_observer: Callable[[Any], None] = None, c_sigma=1, c_intrins=1, render_every_frame=True, **kwargs) \
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
        #agent = (agent+1)%2
        states_transformed = torch.stack(tuple(train_state_rewriter(s) for s in states))
        pols, vals = nets[agent](states_transformed.to(device))
        pollist = F.softmax(pols, dim=-1).tolist()
        vallist = vals.squeeze(-1).tolist()

        return list(zip(pollist, vallist))

    for stat in stats:
                stat.synchronize()

    mcts = MCTSAutAdvList(envs[0].action_space.n, curiosity_evaluator, state_evaluator, curiosity_trainer, c_puct=c_puct,
                aut_stats=stats, c_aut=c_aut, c_sigma=c_sigma, c_intrins=c_intrins, **kwargs)

    def action_value_generator(state, step):
        mcts.mcts_batch(envs, state, num_batches, batch_size)
        probs, values = mcts.get_policy_value(state, 1)
        return probs, list(max(value) for value in values)

    return run_episode_generic_adv(envs, action_value_generator, max_length, max_len_reward,
                               ptan.actions.ProbabilityActionSelector(),
                               state_observer)


#render every frame true
def run_aut_episode(net: torch.nn.Module, env: AutShapingWrapperAdv, max_length: int, max_len_reward: Optional[int],
                    curiosity: ModuleCuriosityOptimizer, device,
                    train_state_rewriter: Callable[[Any], torch.Tensor], stats: AutStats,
                    state_observer: Callable[[Any], None] = None, render_every_frame=True) -> Tuple[
    List[TraceStep], float]:
    stats.synchronize()

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
}


def run():
    torch.multiprocessing.set_start_method("spawn", force=True)
    signal.signal(signal.SIGHUP, throwKeyInterr)

    try:

        cman = CheckpointManager(CHECKPOINT_PATH, LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)
        cman1 = CheckpointManager(CHECKPOINT_PATH_1, LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)

    except EOFError:
        cman = CheckpointManager(CHECKPOINT_PATH + "_copy", LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)
        cman1 = CheckpointManager(CHECKPOINT_PATH_1 + "_copy", LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)


    #TODO: for second agent
    if TRANSPLANT:
        cman = TransplantCheckpointManager(cman, TRANSPLANT_FROM)
        cman.transplant("aut")  # Generating the automaton may not be completely deterministic, we want the same states
    elif OLD_TRANSPLANT:
        cman = TransplantCheckpointManager(cman, TRANSPLANT_FROM)
        for field in OLD_TRANSPLANT:
            cman.transplant(field)


    #should probably do this all the way around, just testing that it works for now,
    # listifying will follow after results
    auts = [
           cman.load("aut", AutomatonSet.from_ltlf(LTLF_SPECS[0], AUT_PARAM_NAMES), PickleLoadHandler()),
           cman1.load("aut1", AutomatonSet.from_ltlf(LTLF_SPECS[1], AUT_PARAM_NAMES), PickleLoadHandler())
           ]

    orig_env = env_constructors[ENV_TYPE](ENV_CONFIG)


    envs = list(
        AutShapingWrapperAdv(orig_env, AUT_PARAM_FUNCS, auts[i], player_num=i, use_potential=False, **AUT_OTHER_PARAMS)
        for i in range(0, len(auts)))


    #quick fix for trivial acceptance states for adversarial automata
    #envs[1].terminate_on_accept = False

##TODO;     we could add this for all nets needed
    #        in a for loop (many agent approach)
    action_spaces = list(env.action_space.n for env in envs)


    writers = [
        SummaryWriter(LOG_FOLDER),
        SummaryWriter(LOG1_FOLDER)
    ]

    train_net_creator, train_rewriter_creator = training_nets[NETWORK]

    nets = [
            cman.load("net", train_net_creator(orig_env, **NETWORK_PARAMS),
                    CombinedLoadHandler(StateDictLoadHandler(), InitZeroLoadHandler())).to(DEVICE),
            cman1.load("net1", train_net_creator(orig_env, ** NETWORK_PARAMS),
                     CombinedLoadHandler(StateDictLoadHandler(), InitZeroLoadHandler())).to(DEVICE)
            ]
    for net in nets:
        #allow inventory channels
        net.share_memory()

    #not using this??
    if CURIOSITY_NET:
        curiosity_net_creator, curiosity_rewriter_creator = curiosity_nets[CURIOSITY_NET]

        icm = cman.load("icm", curiosity_net_creator(orig_env, **CURIOSITY_PARAMS), StateDictLoadHandler()).to(
            DEVICE)
        icm.share_memory()

        icm_opt = cman.load("icm_opt", ModuleCuriosityOptimizer(icm, curiosity_rewriter_creator(orig_env), action_space,
                                                                CURIOSITY_LEARNING_RATE,
                                                                DEVICE), StateDictLoadHandler())
    else:
        icm_opt = NoopCuriosityOptimizer()

    loss_functions = list(loss_funcs[LOSS_FUNC](net=nets[i], device=DEVICE, discount=DISCOUNT, **LOSS_PARAMS)
                          for i in range(0, len(nets)))


    optimizers = [
        cman.load("opt", OPTIMIZER(nets[0].parameters(), lr=LEARNING_RATE, **OPTIMIZER_PARAMS),
                  StateDictLoadHandler()),
        cman1.load("opt1", OPTIMIZER(nets[1].parameters(), lr=LEARNING_RATE, **OPTIMIZER_PARAMS),
                   StateDictLoadHandler())
    ]

    train_rewriter = train_rewriter_creator(orig_env)

    train_loop = cman.load("train_loop",
                           RandomReplayTrainingLoop(DISCOUNT, REPLAY_BUFFER, MIN_TRACE_TO_TRAIN, PPO_TRAIN_ROUNDS,
                                                    train_rewriter, writers, DEVICE),
                           StateDictLoadHandler())
    aut_stats_list = [
        cman.load("aut_stats", AutStats(len(auts[0].graph.network), **AUT_STATS_PARAMS), StateDictLoadHandler()),
        cman1.load("aut_stats1", AutStats(len(auts[1].graph.network), **AUT_STATS_PARAMS), StateDictLoadHandler())
    ]


    #TODO:  transplant multi agent
    if TRANSPLANT:
        orig_alt_stats = cman.load_from_alt("aut_stats", AutStats(len(auts[0].graph.network)), StateDictLoadHandler())
        wrapped_aut_stats = aut_transplant_anneals[ANNEAL_AUT_TRANSPLANT](orig_alt_stats, aut_stats_list[0],
                                                                          **ANNEAL_AUT_TRANSPLANT_PARAMS)
        wrapped_aut_stats.set_step(train_loop.num_rounds)
        train_loop.add_round_hook(wrapped_aut_stats.set_step)
    else:
        wrapped_aut_stats = aut_stats_list

    #same function for both players, just specify playerNum when called
    def aut_hook(trace: List[TraceReturnStep], final_value, playerNum: int):
        # Only count each state once per run
        prev_edges = set()
        last_state = None
        #this was commented TODO when i initially received code: bb912...does it?
        for trst in trace:
        #for trst in reversed(trace):  # TODO does this need to be reversed?
            this_state = frozenset(trst.info["automaton_states"])

            if len(this_state) > 0:
                this_state = set(this_state).pop()
            else:
                this_state = None

            edge = (last_state, this_state)
            last_state = this_state
            if edge[0] is not None and edge[1] is not None:
                if edge not in prev_edges:
                    aut_stats_list[playerNum].visit(edge, trst.discounted_return)
                    prev_edges.add(edge)



    #use only 1 train_loop, specify aut hooks for each agent. array of hooks
    train_loop.add_trace_hook(aut_hook)


    with get_parallel_queue_adv(num_processes=NUM_PROCESSES, episode_runner=episode_runners[EPISODE_RUNNER_TYPE],
                            nets=nets, envs=envs, max_length=MAX_EPISODE_LEN, max_len_reward=MAX_LEN_REWARD,
                            curiosity=icm_opt, state_observer=None, device=DEVICE,
                            stats=wrapped_aut_stats, train_state_rewriter=train_rewriter,
                            **EPISODE_RUNNER_PARAMS) as sim_round_queue:
        # random.seed(798)

        while True:

            train_loop(sim_round_queue, loss_functions, optimizers)

            if train_loop.num_rounds % CHECKPOINT_EVERY == 0:
                #print("num_rounds=", train_loop.num_rounds)

                save_dict = {
                    "net": nets[0],
                    "opt": optimizers[0],
                    "train_loop": train_loop,
                    "aut_stats": aut_stats_list[0],
                    "aut": auts[0],
                }
                save_dict1 = {
                    "net1": nets[1],
                    "opt1": optimizers[1],
                    "aut_stats1": aut_stats_list[1],
                    "aut1": auts[1],
                }

                if CURIOSITY_NET:
                    save_dict.update({
                        "icm": icm,
                        "icm_opt": icm_opt
                    })
                cman.save(save_dict)
                cman1.save(save_dict1)

                if STOP_AFTER and train_loop.global_step > STOP_AFTER:
                    print("STOPPING: step limit " + str(train_loop.global_step) + "/" + str(STOP_AFTER))
                    break


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run()
