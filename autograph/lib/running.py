import functools
import multiprocessing.context
import os
import random
from _signal import SIGKILL
from collections import deque
from copy import deepcopy
from typing import Callable, Any, Tuple, List, Union

import numpy
import psutil
import ptan
import torch
from decorator import contextmanager
from gym import Env
from tensorboardX import SummaryWriter
from torch import multiprocessing as multiprocessing
from torch.nn import functional as F
from torch.optim import Optimizer

from autograph.lib.envs.saveloadenv import SaveLoadEnv
from autograph.lib.loss_functions import LossFunction
from autograph.lib.mcts import MCTS
from autograph.lib.mcts_aut_adv_team import MCTSAutAdvList
from autograph.lib.shaping import AutShapingWrapperAdv
from autograph.lib.util.trace_info_processor import TraceInfoPreprocessor
from autograph.lib.util.trace_return_step import TraceStep, TraceReturnStep
from autograph.net.curiosity.curiosity_optimizer import ModuleCuriosityOptimizer
from autograph.net.mazenet import Mazenet


# TODO Run this directly on CPU because it isn't a batch?
def run_episode_generic(env: Env, action_value_generator: Callable[[Any, int], Tuple[List[float], float]],
                        max_length: int,
                        max_len_reward: Union[int, None],
                        action_selector: ptan.actions.ActionSelector = ptan.actions.ArgmaxActionSelector(),
                        state_observer: Callable[[Any], None] = None):
    """
    Run an episode in an environment
    :param env: The environment to run the episode in
    :param action_value_generator: Given a state and step number, produce some action probabilities and a value estimate
    :param max_length: The step number after which to cut off the episode
    :param action_selector: How to decide the right actions to select given NN "probabilities"
    :param state_observer:A callback function to notify of all states encountered
    :return: A list of steps taken, and the value estimate of the final state
    """
    done = False

    # changed this reset function to not go to initial board state
    next_state = env.reset()
    # next_state = env
    next_action_raw, next_value = action_value_generator(next_state, 0)
    length = 0

    trace = []

    while not done:
        length += 1

        # Take an action and get results of the action
        state, action_raw, value = next_state, next_action_raw, next_value
        # action_probs = F.softmax(action_raw)
        # action_selected = action_selector(action_probs.unsqueeze(0).cpu().numpy())[0]
        # TODO
        action_selected = action_selector(numpy.array([action_raw]))[0]
        next_state, reward, done, info = env.step(action_selected)

        # what to do with the info from stepping nonTurn autonama

        if state_observer:
            state_observer(next_state)

        done_from_env = 1 if done else 0

        # Get the next action/value pair (now instead of beginning of loop so that we easily have access to next value)
        if done:
            next_value = 0
        else:
            # Performs an MCTS tree search and then returns pi_play...
            next_action_raw, next_value = action_value_generator(next_state, length)
            if length >= max_length:
                done = True
                if max_len_reward is not None:
                    next_value = max_len_reward

        trace.append(
            TraceStep(state, value, action_raw, action_selected, next_state, next_value, reward, info, done_from_env))

    return trace, next_value


def run_episode_generic_adv(envs: List[Env], action_value_generator: Callable[[Any, int], Tuple[List[float], float]],
                            max_length: int,
                            max_len_reward: Union[int, None],
                            action_selector: ptan.actions.ActionSelector = ptan.actions.ArgmaxActionSelector(),
                            state_observer: Callable[[Any], None] = None,
                            show_play_step: bool = True,
                            hide_and_seek: bool = True):
    """
    Run an episode in an environment
    :param env: The environment to run the episode in
    :param action_value_generator: Given a state and step number, produce some action probabilities and a value estimate
    :param max_length: The step number after which to cut off the episode
    :param action_selector: How to decide the right actions to select given NN "probabilities"
    :param state_observer:A callback function to notify of all states encountered
    :return: A list of steps taken, and the value estimate of the final state
    """
    done_from_env = 0

    # this reset function seems to be keeping us near the beginning of traces when we print
    # playsteps
    next_states = list(env.reset() for env in envs)

    turnn = next_states[0][3]

    next_state = next_states[turnn]
    next_full_state = envs[0].env.get_obs_full()

    next_action_raw, next_values = action_value_generator(next_state, 0)
    length = 0

    if show_play_step:
        print("play step at" + str(length))
        envs[0].render()
    trace = []

    traces = [trace[:] for i in range(0, len(next_state[0]))]

    while done_from_env == 0:
        length += 1
        full_state = next_full_state
        # Take an action and get results of the action
        state, action_raw, values = next_state, next_action_raw, next_values
        turnn = state[3]
        # action_probs = F.softmax(action_raw)
        # action_selected = action_selector(action_probs.unsqueeze(0).cpu().numpy())[0]
        # TODO
        action_selected = action_selector(numpy.array([action_raw]))[0]

        # Hide and Seek functionality
        # if hide_and_seek and turnn % 2 == 1 and length < 200:
        #    action_selected = 4

        rewards = [0] * len(state[0])
        dones = [bool] * len(state[0])
        infos = [None] * len(state[0])

        for i in range(0, len(state[0])):

            next_state, rewards[i], dones[i], infos[i] = envs[i].step(action_selected)

            # what to do with step info from other autonoma?
            if i == 0:
                next_full_state = envs[0].env.get_obs_full()

        if show_play_step:
            print("play step at" + str(length))
            envs[0].render()

        if state_observer:
            state_observer(next_state)

        done_from_env = 0
        for done in dones:
            if done:
                done_from_env = 1

        # Get the next action/value pair (now instead of beginning of loop so that we easily have access to next value)
        if done_from_env == 1:
            for i in range(0, len(next_values)):
                next_values[i] = 0
        else:
            turnn = (turnn + 1) % 2
            # Performs an MCTS tree search and then returns pi_play
            next_action_raw, next_values = action_value_generator(next_state, turnn)
            if length >= max_length:
                done_from_env = 1

                # hiders get a reward if we reach the length
                if max_len_reward is not None:
                    if hide_and_seek:
                        next_values[0] = max_len_reward
                        next_values[2] = max_len_reward

        for agent in range(0, len(state[0])):
            # each agent gets their own traces array to train on , with their specific values

            traces[agent].append(TraceStep(full_state, values[agent], action_raw, action_selected,
                                           next_full_state, next_values[agent], rewards[agent], infos[agent],
                                           1 if dones[agent] else 0))
        x = 1
    return traces, next_values


def calculate_elo(r1: int, r2: int, S: float):
    # win can be 0, 0.5, or 1
    # 0 represents loss, 0.5 represents draw, 1 represents win
    R1 = 10 ** (r1 / 400)
    R2 = 10 ** (r2 / 400)
    E1 = R1 / (R1 + R2)
    elo1 = r1 + 32 * (S - E1)
    return elo1


def run_episode_generic_team(envs: List[AutShapingWrapperAdv],
                             action_value_generator: Callable[[Any, int], Tuple[List[float], float]],
                             max_length: int,
                             max_len_reward: Union[int, None],
                             action_selector: ptan.actions.ActionSelector = ptan.actions.ArgmaxActionSelector(),
                             state_observer: Callable[[Any], None] = None,
                             show_play_step: bool = True,
                             hide_and_seek: bool = True):
    """
    Run an episode in an environment
    :param env: The environment to run the episode in
    :param action_value_generator: Given a state and step number, produce some action probabilities and a value estimate
    :param max_length: The step number after which to cut off the episode
    :param action_selector: How to decide the right actions to select given NN "probabilities"
    :param state_observer:A callback function to notify of all states encountered
    :return: A list of steps taken, and the value estimate of the final state
    """
    done_from_env = 0

    # this reset function seems to be keeping us near the beginning of traces when we print
    # playsteps
    next_states = list(env.reset() for env in envs)

    turnn = next_states[0][3]
    team_turn = turnn % len(envs)

    next_state = next_states[team_turn]
    next_full_state = envs[0].env.get_obs_full()

    next_action_raw, next_values = action_value_generator(next_state, 0)
    length = 0

    if show_play_step:
        print("play step at" + str(length))
        envs[0].render()
    trace = []

    traces = [trace[:] for i in range(0, len(envs))]

    while done_from_env == 0:
        length += 1
        full_state = next_full_state
        # Take an action and get results of the action
        state, action_raw, values = next_state, next_action_raw, next_values
        turnn = state[3]
        # action_probs = F.softmax(action_raw)
        # action_selected = action_selector(action_probs.unsqueeze(0).cpu().numpy())[0]
        # TODO
        action_selected = action_selector(numpy.array([action_raw]))[0]

        # Hide and Seek functionality
        # if hide_and_seek and turnn % 2 == 1 and length < 200:
        #    action_selected = 4
        if show_play_step:
            print("play step at" + str(length))
            print("agent" + str(turnn) + "doing action" + str(action_selected))
            envs[turnn % len(envs)].render()

        rewards = [0] * len(envs)
        dones = [bool] * len(envs)
        infos = [None] * len(envs)

        # once per team, gather information from action selected
        for i in range(0, len(envs)):

            next_state, rewards[i], dones[i], infos[i] = envs[i].step(action_selected)

            if i == 0:
                next_full_state = envs[0].env.get_obs_full()

        if state_observer:
            state_observer(next_state)

        # if one team wins (done from their env) give all other teams their terminate on fail reward
        done_from_env = 0
        for idx, done in enumerate(dones):
            if done:
                done_from_env = 1
                next_values[idx] = 1
                for jdx in range(0, len(rewards)):
                    if jdx != idx:
                        rewards[jdx] += envs[jdx].termination_fail_reward
                        next_values[jdx] = rewards[jdx]

        # Get the next action/value pair (now instead of beginning of loop so that we easily have access to next value)
        if done_from_env == 1:
            pass
        else:
            turnn = (turnn + 1) % len(envs[0].env.positions)

            # Performs an MCTS tree search and then returns pi_play.... the turnn doesn't do anything
            next_action_raw, next_values = action_value_generator(next_state, turnn)
            if length >= max_length:
                done_from_env = 1
                if max_len_reward is not None:
                    for rew in range(0, len(rewards)):
                        rewards[rew] = max_len_reward
                        next_values[rew] = max_len_reward

        # again, we save traces and values once per team
        for agent in range(0, len(envs)):
            # each agent gets their own traces array to train on , with their specific values
            x = full_state

            traces[agent].append(TraceStep(full_state, values[agent], action_raw, action_selected,
                                           next_full_state, next_values[agent], rewards[agent], infos[agent],
                                           1 if done_from_env else 0))

    return traces, next_values


def run_episode_generic_team_cnn_only(envs: List[AutShapingWrapperAdv],
                                      get_pi_cnn: Callable[[Any, int], Tuple[List[float], float]],
                                      max_length: int,
                                      max_len_reward: Union[int, None],
                                      action_selector: ptan.actions.ActionSelector = ptan.actions.ArgmaxActionSelector(),
                                      state_observer: Callable[[Any], None] = None,
                                      show_play_step: bool = True,
                                      hide_and_seek: bool = True):
    """
    Run an episode in an environment
    :param env: The environment to run the episode in
    :param action_value_generator: Given a state and step number, produce some action probabilities and a value estimate
    :param max_length: The step number after which to cut off the episode
    :param action_selector: How to decide the right actions to select given NN "probabilities"
    :param state_observer:A callback function to notify of all states encountered
    :return: A list of steps taken, and the value estimate of the final state
    """
    done_from_env = 0

    # this reset function seems to be keeping us near the beginning of traces when we print
    # playsteps
    next_states = list(env.reset() for env in envs)

    turnn = next_states[0][3]
    team_turn = turnn % len(envs)

    next_state = next_states[team_turn]

    next_action_raw = get_pi_cnn(next_state, team_turn)
    length = 0

    if show_play_step:
        print("play step at" + str(length))
        envs[0].render()
    trace = []

    traces = [trace[:] for i in range(0, len(envs))]

    while done_from_env == 0:
        length += 1

        # Take an action and get results of the action
        state, action_raw = next_state, next_action_raw
        turnn = state[3]
        # action_probs = F.softmax(action_raw)
        # action_selected = action_selector(action_probs.unsqueeze(0).cpu().numpy())[0]
        # TODO
        action_selected = action_selector(numpy.array([action_raw]))[0]

        # Hide and Seek functionality
        # if hide_and_seek and turnn % 2 == 1 and length < 200:
        #    action_selected = 4

        rewards = [0] * len(envs)
        dones = [bool] * len(envs)
        infos = [None] * len(envs)

        # once per team, gather information from action selected
        for i in range(0, len(envs)):
            next_state, rewards[i], dones[i], infos[i] = envs[i].step(action_selected)

        if show_play_step:
            print("play step at" + str(length))
            print("agent" + str(turnn) + "doing action" + str(action_selected))
            envs[turnn % len(envs)].render()

        if state_observer:
            state_observer(next_state)

        # if one team wins (done from their env) give all other teams their terminate on fail reward
        done_from_env = 0
        for idx, done in enumerate(dones):
            if done:
                done_from_env = 1
                for jdx in range(0, len(rewards)):
                    if jdx != idx:
                        rewards[jdx] += envs[jdx].termination_fail_reward

        # Get the next action/value pair (now instead of beginning of loop so that we easily have access to next value)
        if done_from_env == 1:
            pass
        else:
            turnn = (turnn + 1) % len(envs[0].env.positions)

            # Performs an MCTS tree search and then returns pi_play.... the turnn doesn't do anything
            next_action_raw = get_pi_cnn(next_state, turnn % len(envs))
            if length >= max_length:
                done_from_env = 1
                if max_len_reward is not None:
                    for rew in range(0, len(rewards)):
                        rewards[rew] = max_len_reward

    return traces, [0, 0]


def run_episode(net: Mazenet, env: Env, max_length: int, device,
                action_selector: ptan.actions.ActionSelector = ptan.actions.ArgmaxActionSelector(),
                state_observer: Callable[[Any], None] = None) -> Tuple[List[TraceStep], float]:
    """
    Run an episode using the policy from the given network
    :param net: The network whose policy to follow- should have a forward_obs method that returns the action outputs and
    value estimate given a single observation
    :param env: The environment to run the simulation in
    :param max_length: How long the episode should be allowed to run before cutting it off
    :param action_selector: A callable that accepts a list of action probabilities and returns the action to choose
    :param state_observer: A callable that will be called with each state
    :param device: What device to run the network on
    :return: A list of TraceStep tuples and a value estimate of the last state
    """

    def obs_helper(state, step):
        action, value = net.forward_obs(state, device)
        return action.detach(), value.detach().squeeze(-1)

    return run_episode_generic(env, obs_helper, max_length, action_selector, state_observer)


def run_mcts_episode(net: Mazenet, env: SaveLoadEnv, max_length: int, curiosity: ModuleCuriosityOptimizer, device,
                     c_puct,
                     num_batches, batch_size,
                     state_observer: Callable[[Any], None] = None) -> Tuple[List[TraceStep], float]:
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
        states, actions, rewards, next_states = zip(*sars)
        rewards = curiosity.get_curiosity(states, actions, next_states)
        return rewards.tolist()

    def curiosity_trainer(sars):
        states, actions, rewards, next_states = zip(*sars)
        curiosity.train(states, actions, next_states, train_rounds=1)

    def state_evaluator(states):
        states_transformed = torch.stack(tuple(net.rewrite_obs(s) for s in states))
        pols, vals = net(states_transformed.to(device))
        pollist = F.softmax(pols, dim=-1).tolist()
        vallist = vals.squeeze(-1).tolist()

        return list(zip(pollist, vallist))

    mcts = MCTS(env.action_space.n, curiosity_evaluator, state_evaluator, curiosity_trainer, c_puct=c_puct)

    def action_value_generator(state, step):
        mcts.mcts_batch(env, state, num_batches, batch_size)
        probs, values = mcts.get_policy_value(state, 1)
        return probs, max(values)

    return run_episode_generic(env, action_value_generator, max_length, ptan.actions.ProbabilityActionSelector(),
                               state_observer)


def calculate_returns_adv(trace: List[TraceStep], last_value: float, discount: float) \
        -> List[Tuple[float, float, float]]:
    """
    Given a trace, get the discounted returns, advantage, and return advantage value. Advantage is calculated based on the actual
    discounted return versus the value estimate of the state.
    :param trace: The information from a single run
    :param last_value: A value estimate of the final state
    :param discount: The discount factor
    :return: A list of (discounted return, advantage, return_advantage) tuples
    """
    discounted_return = last_value
    next_value = last_value
    returns_adv = []

    for step in reversed(trace):
        discounted_return *= discount
        discounted_return += step.reward
        advantage = step.reward + (discount * next_value) - step.value
        return_advantage = discounted_return - float(step.value)
        returns_adv.append((discounted_return, advantage, return_advantage))

        next_value = step.value

    returns_adv.reverse()

    return returns_adv


def parallel_queue_worker(queue: multiprocessing.SimpleQueue, function_to_run: Callable) -> None:
    """
    Worker that repeatedly adds the result of a function to a queue, but eventually quits when the parent process dies.
    :param queue: The queue to add results to
    :param function_to_run: A function that takes no arguments and produces no results
    :return: When the parent process dies
    """
    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()

    # random.seed(123)

    while True:
        result = function_to_run()

        # If the parent process isn't python, that means that it got terminated
        # (and this orphan process got reassigned to init or to some display manager)
        parent = psutil.Process(os.getppid())
        if "python" not in parent.name():
            return

        queue.put(result)
        # pr.dump_stats("performance.cprof")


class RenderEnvAndReturnWrapper:
    """
    Wraps a given callable such that if the callable accepts an env as a parameter, the env is printed before returning.

    Not a nested function for pickling reasons.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, envs: List[Env], **kwargs):
        ret = self.func(envs=envs, **kwargs)
        # for env in envs:
        #   env.env.load_state(ret[0][0].state)
        envs[ret[0][0][0].state[3]].render()
        return ret


@contextmanager
def get_parallel_queue_adv(num_processes, episode_runner, envs, **kwargs):
    """
    Create a queue that has a bunch of parallel feeder processes
    :param num_processes: How many feeder processes
    :param episode_runner: A function that produces a trace to be added to the queue
    :param envs: The environments (autEnvWrappers) to run the simulations in
    :param kwargs: Additional arguments to the episode runner
    :return: The queue that the processes are feeding into, and a list of the created processes
    """
    multiprocessing.set_start_method("spawn", True)

    sim_round_queue = multiprocessing.SimpleQueue()
    processes: List[multiprocessing.context.Process] = []
    for i in range(num_processes):
        newenvs = list(deepcopy(env) for env in envs)
        render = (i == 0)
        render = False

        if render:
            this_episode_runner = RenderEnvAndReturnWrapper(episode_runner)
        else:
            this_episode_runner = episode_runner

        mcts_with_args = functools.partial(this_episode_runner, envs=newenvs, **kwargs)

        p = multiprocessing.Process(target=parallel_queue_worker,
                                    args=(sim_round_queue, mcts_with_args))
        p.daemon = True
        p.name = "Worker thread " + str(i)
        p.start()
        processes.append(p)

    try:
        yield sim_round_queue
    finally:
        for p in processes:
            os.kill(p.pid, SIGKILL)


@contextmanager
def get_parallel_queue(num_processes, episode_runner, env, **kwargs):
    """
    Create a queue that has a bunch of parallel feeder processes
    :param num_processes: How many feeder processes
    :param episode_runner: A function that produces a trace to be added to the queue
    :param env: The environment to run the simulations in
    :param kwargs: Additional arguments to the episode runner
    :return: The queue that the processes are feeding into, and a list of the created processes
    """
    multiprocessing.set_start_method("spawn", True)

    sim_round_queue = multiprocessing.SimpleQueue()
    processes: List[multiprocessing.context.Process] = []
    for i in range(num_processes):
        newenv = deepcopy(env)
        render = (i == 0)

        if render:
            this_episode_runner = RenderEnvAndReturnWrapper(episode_runner)
        else:
            this_episode_runner = episode_runner

        mcts_with_args = functools.partial(this_episode_runner, env=newenv, **kwargs)

        p = multiprocessing.Process(target=parallel_queue_worker,
                                    args=(sim_round_queue, mcts_with_args))
        p.daemon = True
        p.name = "Worker thread " + str(i)
        p.start()
        processes.append(p)

    try:
        yield sim_round_queue
    finally:
        for p in processes:
            os.kill(p.pid, SIGKILL)


class RandomReplayTrainingLoop:
    """
    A training loop that reads random replays from a replay buffer and trains a loss function based on that
    """

    def __init__(self, discount: float, replay_buffer_len: int, min_trace_to_train: int, train_rounds: int,
                 obs_processor: Callable, writers: List[SummaryWriter], device):
        self.device = device
        self.obs_processor = obs_processor
        self.train_rounds = train_rounds
        self.min_trace_to_train = min_trace_to_train
        self.discount = discount
        self.writers = writers

        self.trace_hooks: List[Callable[[List[TraceReturnStep], float], None]] = []
        self.round_hooks: List[Callable[[int], None]] = []

        self.recent_traces: List[deque[TraceReturnStep]] = [deque(maxlen=replay_buffer_len).copy()] * len(writers)

        for i in range(0, len(writers)):
            self.recent_traces[i] = deque(maxlen=replay_buffer_len)

        self.global_step = 0
        self.num_rounds = 0
        self.elo_ratings = [1200, 1200]

    def add_trace_hook(self, hook: Callable[[List[TraceReturnStep], float], None]):
        self.trace_hooks.append(hook)

    def add_round_hook(self, hook: Callable[[int], None]):
        self.round_hooks.append(hook)

    # player number refers to team number when sharing nets between teams
    def process_trace(self, trace: List[TraceStep], last_val: float, player_num: int):
        """
        Calculate the returns on a trace and add it to the replay buffer
        :param trace: The actions actually taken
        :param last_val: The value estimate of the final state
        """
        ret_adv = calculate_returns_adv(trace, last_val, discount=self.discount)

        trace_adv = [TraceReturnStep(*twi, *ra) for twi, ra in zip(trace, ret_adv)]

        for hook in self.trace_hooks:
            hook(trace_adv, last_val, player_num)

        self.recent_traces[player_num].extend(trace_adv)

        self.writers[player_num].add_scalar("run/total_reward", sum([step.reward for step in trace]), self.global_step)
        self.writers[player_num].add_scalar("run/length", len(trace), self.global_step)

        # we don't need this to go through every trace and add?
        for orig_step, new_step in zip(trace, trace_adv):
            self.global_step += 1 if player_num == 0 else 0
            # self.writer.add_scalar("reward/external", orig_step.reward, self.global_step)
            # self.writer.add_scalar("advantage", new_step.advantage, self.global_step)

    def train_on_traces(self, traces: List[TraceReturnStep],
                        loss_function: LossFunction,
                        optimizer: Optimizer):
        """
        Minimize a loss function based on a given set of replay steps.
        :param traces:
        :return:
        """
        trinfo = TraceInfoPreprocessor(traces, self.obs_processor, self.device)

        loss, logs = loss_function(trinfo)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(None)

        logs["total_loss"] = loss

        return logs

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "recent_traces": self.recent_traces,
            "num_rounds": self.num_rounds,
            "elo_ratings": self.elo_ratings
        }

    def load_state_dict(self, state):
        renames = {
            "globalstep": "global_step",
            "numrounds": "num_rounds",
            "eloratings": "elo_ratings"
        }

        for key, value in renames.items():
            if key not in state and key is "eloratings":
                state["elo_ratings"] = [1200, 1200]

            if key in state and value not in state:
                state[value] = state[key]

        self.global_step = state["global_step"]
        self.num_rounds = state["num_rounds"]
        self.recent_traces = state["recent_traces"]
        self.elo_ratings = state["elo_ratings"]

    def __call__(self, sim_round_queue, loss_functions, optimizers):
        self.num_rounds += 1

        traces, last_values = sim_round_queue.get()

        max_val = 0
        winner_idx = 0
        for i in range(0, len(last_values)):
            if last_values[i] > max_val:
                max_val = last_values[i]
                winner_idx = i
        for i in range(0, len(last_values)):
            # calculate elo of all teams in comparison with the winning team
            if i != winner_idx:
                r1 = self.elo_ratings[winner_idx]
                r2 = self.elo_ratings[i]
                if max_val != 0:
                    self.elo_ratings[winner_idx] = calculate_elo(r1, r2, 1)
                    self.elo_ratings[i] = calculate_elo(r2, r1, 0)
                else:
                    # if max_val is 0 the game is tied
                    self.elo_ratings[winner_idx] = calculate_elo(r1, r2, 0.5)
                    self.elo_ratings[i] = calculate_elo(r2, r1, 0.5)
        for i in range(0, len(traces)):
            self.process_trace(traces[i], last_values[i], i)
        # self.process_trace(trace, last_values)

        for i in range(0, len(traces)):
            if len(self.recent_traces[i]) < self.min_trace_to_train:
                return

        # this agent in the nested for loop refers to team when we are sharing nets
        for i in range(self.train_rounds):
            for agent in range(0, len(traces)):
                rand_traces = random.sample(self.recent_traces[agent], self.min_trace_to_train)
                logs = self.train_on_traces(rand_traces, loss_functions[agent], optimizers[agent])

                if i == self.train_rounds - 1:
                    for key, value in logs.items():
                        self.writers[agent].add_scalar(key, value, self.global_step)

        for agent in range(0, len(last_values)):
            self.writers[agent].add_scalar("run/elo_rating", self.elo_ratings[agent], self.global_step)

        for hook in self.round_hooks:
            hook(self.num_rounds)
