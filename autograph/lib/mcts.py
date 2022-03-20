"""
Monte-Carlo Tree Search
Adapted from Deep Reinforcement Learning Hands On Chapter 18
"""
import math
from typing import Dict, Callable, Any, Tuple, List, TypeVar, Generic
from typing import Sequence

import numpy as np
from gym import Env

# Good article about parameters:
# https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
from autograph.lib.envs.saveloadenv import SaveLoadEnv
from autograph.lib.tree.max_tree_operation import MaxTreeOperation
from autograph.lib.tree.sigma_tree_operation import SigmaTreeOperation
from autograph.lib.tree.tree_operation import TreeOperation
from autograph.lib.util import StatsTracker

T = TypeVar("T")
INTRINS_CONSTANT = 10

BackupElement = Tuple[T, int, int, T]
BackupTrace = List[BackupElement]


class MCTS(Generic[T]):
    def __init__(self, num_actions: int,
                 curiosity_evaluator: Callable[[List[BackupElement]], List[int]],
                 state_evaluator: Callable[[List[T]], List[Tuple[Sequence[float], float]]],
                 curiosity_trainer: Callable[[List[BackupElement]], None],
                 c_puct: int = 1.0, mcts_to_completion=False, discount=1):
        """
        :param c_puct: How much to weight the randomness and the NN-calculated probabilities as opposed to values
        :param mcts_to_completion: Keep traversing tree until end of episode, otherwise stop when we reach a leaf and
        use the value estimate as the reward function
        :param curiosity_trainer: A function that accepts a list of (state, action) pairs and uses this
        information to train the curiosity metric that these states have been visited
        :param state_evaluator: Given a list of states, give back a list of (action probabilities, value estimate)
        :param curiosity_evaluator: List of (state, action) pairs to list of raw curiosity values
        """

        self.mcts_to_completion = mcts_to_completion
        self.curiosity_evaluator = curiosity_evaluator
        self.state_evaluator = state_evaluator
        self.curiosity_trainer = curiosity_trainer
        self.num_actions = num_actions
        self.discount = discount

        self.c_puct = c_puct
        # count of visits, state_int -> [N(s, a)]
        self.visit_count = {}
        # total value of the state's action, state_int -> [W(s, a)]
        self.value = {}
        # average value of actions, state_int -> [Q(s, a)]
        self.value_avg = {}
        # prior probability of actions, state_int -> [P(s,a)]
        self.probs = {}
        # most recently computed intrinsic reward for single action step, state -> [IR(s, a)]
        self.intrinsic = {}
        # tree uncertainty https://arxiv.org/pdf/1805.09218.pdf
        self.sigma = {}

        self.intrinsic_maxchildren = {}

        self.intrinsic_stats = StatsTracker()

        # The properties to backup during the backwards MCTS pass, along with how to actually do the backup
        self.tree_backups: List[Tuple[Dict[Any, List[float]], TreeOperation]] = [
            (self.intrinsic, MaxTreeOperation()),
            (self.sigma, SigmaTreeOperation()),
        ]

    def clear(self) -> None:
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.probs.clear()
        self.sigma.clear()
        self.intrinsic.clear()

    def __len__(self) -> int:
        return len(self.value)

    def pick_action(self, state: T, root_state: bool) -> int:

        counts = self.visit_count[state]
        total_count_sqrt = max(math.sqrt(sum(counts)), 1)
        probs = self.probs[state]
        values_avg = self.value_avg[state]
        intrins = self.intrinsic_maxchildren[state]
        intrins = [i if i is not None else self.intrinsic_stats.average() for i in intrins]
        total_intrins = sum(intrins)
        if total_intrins == 0:
            intrins_normalized = np.zeros_like(intrins)
        else:
            intrins_normalized = [i / total_intrins for i in intrins]

        sigma = self.sigma[state]

        if root_state:
            # Add random noise to (NN-calculated) probabilities
            noises = np.random.dirichlet([0.03] * self.num_actions)
            probs = [0.75 * prob + 0.25 * noise for prob, noise in zip(probs, noises)]

        # Calculate score based on
        score = [value + self.c_puct * prob * (total_count_sqrt / (1 + count)) * sigma * intrin
                 for value, prob, intrin, count, sigma in zip(values_avg, probs, intrins_normalized, counts, sigma)]
        action = int(np.argmax(score))
        return action

    def find_leaf(self, env: Env, cur_state: T) -> Tuple[BackupTrace, T, bool]:
        """
        Starting from the root state, traverse MCTS tree until either a leaf is found or a terminal state is reached
        :param env: The environment to run the simulation in
        :param cur_state: The root state
        :return: Trajectory taken, last state, is done
        """
        turns = []  # Tuple of state, action, reward, next state
        seenstates = {cur_state}

        done = False
        root = True
        # TODO only allow the search to select allowed moves
        while not done and not self.is_leaf(cur_state):
            action = self.pick_action(cur_state, root)
            root = False

            next_state, reward, done, _ = env.step(action)
            turns.append((cur_state, action, reward, next_state))

            if done or next_state in seenstates:
                done = True
                self.sigma[cur_state][action] = 0
            else:
                seenstates.add(next_state)

            cur_state = next_state

        return turns, cur_state, done

    def find_leaf_batch(self, env: SaveLoadEnv, num_leaves: int, start_state: T) \
            -> Tuple[List[BackupElement], List[BackupTrace], List[Tuple[BackupTrace, T]]]:
        """
        Run multiple MCTS simulations and aggregate some of the results
        :param env: Copyable environment
        :param num_leaves: How many times to run the simulation
        :param start_state: Root node for search
        :return: States that need curiosity values, state-action pairs that should be used to train curiosity,
                 a list of action trajectories (for terminal runs), and a list of trajectory-value-estimate pairs for
                 non-terminal runs
        """

        intrinsic_to_calculate = set()
        will_create = set()
        backup_queue = []
        create_queue = []

        env_state = env.save_state()

        for i in range(num_leaves):
            turns, end_state, done = self.find_leaf(env, start_state)

            intrinsic_to_calculate.update(turns)

            if done:
                backup_queue.append(turns)
            else:
                if end_state not in will_create:
                    will_create.add(end_state)
                    create_queue.append((turns, end_state))

            env.load_state(env_state)  # Reset environment

        return list(intrinsic_to_calculate), backup_queue, create_queue

    def update_curiosity(self, states: List[BackupElement]) -> None:
        """
        Use the curiosity evaluator to update intrinsic rewards
        :param states: The states to update the intrinsic rewards in
        """
        curiosity_values = self.curiosity_evaluator(states)

        for i, (state, action, reward, next_state) in enumerate(states):
            if self.intrinsic[state][action] is not None:
                self.intrinsic_stats.remove_point(self.intrinsic[state][action])
            self.intrinsic[state][action] = curiosity_values[i]
            self.intrinsic_maxchildren[state][action] = curiosity_values[i]
            self.intrinsic_stats.add_point(curiosity_values[i])

    def create_state(self, cur_state: T, probs: Sequence[float]) -> None:
        """
        Create an empty state
        :param cur_state: The state to create
        :param probs: The action probabilities for the state
        """
        num_actions = self.num_actions

        self.visit_count[cur_state] = [0] * num_actions
        self.value[cur_state] = [0.0] * num_actions
        self.value_avg[cur_state] = [0.0] * num_actions
        self.probs[cur_state] = probs
        self.intrinsic[cur_state] = [None] * num_actions
        self.intrinsic_maxchildren[cur_state] = [None] * num_actions
        self.sigma[cur_state] = [1.0] * num_actions

    def mcts_mini_batch(self, env: SaveLoadEnv, start_state: T, batch_size: int) -> None:
        """
        Run a minibatch of mcts and update curiosity metrics
        :param env: Environment to run the minibatch in. Must be copyable
        :param start_state: Root state to start search from
        :param batch_size: How many MCTS simulations to take
        """
        intrinsic_calculate, backups, creates = self.find_leaf_batch(env,
                                                                     start_state=start_state,
                                                                     num_leaves=batch_size)
        if len(creates) > 0:
            create_trajectories, create_states = zip(*creates)
            state_infos = self.state_evaluator(create_states)
        else:
            create_trajectories, create_states = [], []
            state_infos = []

        backups_with_final_values = []
        for bup in backups:
            backups_with_final_values.append((bup, 0))

        for state, bup, (policy, value) in zip(create_states, create_trajectories, state_infos):
            self.create_state(state, policy)
            backups_with_final_values.append((bup, float(value)))

        if len(intrinsic_calculate) > 0:
            self.update_curiosity(intrinsic_calculate)
            self.curiosity_trainer(intrinsic_calculate)

        for bup, value in backups_with_final_values:
            self.backup_mcts_trace(bup, value)

    def is_leaf(self, state: T) -> bool:
        """
        Is the given state not expanded yet?
        """
        return state not in self.probs

    def backup_mcts_trace(self, sars: BackupTrace, final_value_estimate: float = 0.0) -> float:
        """
        Given a trace of a MCTS forward run, use the final value estimates and the actual rewards to backup the
        tree node values
        :param sars: List of tuples of state, action, reward
        :param final_value_estimate: A final value estimate if the MCTS doesn't reach the terminal state.
        If the MCTS runs a simulation until termination, this should be 0.
        :return The value of the first state
        """
        value_discounted = final_value_estimate

        prev_tree_values = [None] * len(self.tree_backups)

        for state, action, reward, next_state in reversed(sars):
            # Backup values according to the tree operations defined in the constructor
            for idx, ((storage, treeop), prev_value) in enumerate(zip(self.tree_backups, prev_tree_values)):
                if prev_value is not None:
                    storage[state][action] = treeop.edge_combinator(storage[state][action], prev_value)

                prev_tree_values[idx] = treeop.node_value(storage[state])

            value_discounted *= self.discount
            value_discounted += reward

            self.visit_count[state][action] += 1
            self.value[state][action] += value_discounted
            self.value_avg[state][action] = self.value[state][action] / self.visit_count[state][action]

        return value_discounted

    def mcts_batch(self, env: SaveLoadEnv, state: Any, count: int, batch_size: int) -> None:
        """
        Run a batch of MCTS
        :param env: The environment to run the batch in. Must be copyable
        :param state: The state we are currently in
        :param count: How many minibatches to run
        :param batch_size: Size of each minibatch
        """
        for _ in range(count):
            self.mcts_mini_batch(env, state, batch_size)

    def get_policy_value(self, state: T, tau: float = 1) -> Tuple[List[float], List[float]]:
        """
        Extract policy and action-values by the state
        :param state: state of the board
        :param tau: temperature (as defined in alphago zero paper)
        :return: (probs, values)
        """
        counts = self.visit_count[state]
        if tau == 0:  # Set the best action to 1, others at 0
            probs = [0.0] * len(counts)
            probs[np.argmax(counts).item()] = 1.0
        else:  # Make the best action stand out more or less
            counts = [count ** (1.0 / tau) for count in counts]
            total = sum(counts)
            probs = [count / total for count in counts]
        values = self.value_avg[state]
        return probs, values
