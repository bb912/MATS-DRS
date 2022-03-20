"""
Monte-Carlo Tree Search
Adapted from Deep Reinforcement Learning Hands On Chapter 18
"""
import math
from abc import ABCMeta, abstractmethod
from typing import Dict, Callable, Any, Tuple, List, TypeVar, Generic, Set, Union, FrozenSet
from typing import Sequence

import numpy as np
from gym import Env
# Good article about parameters:
# https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
from torch.multiprocessing import Array, Lock, Value

from autograph.lib.shaping import AutShapingWrapper
from autograph.lib.tree.max_tree_operation import MaxTreeOperation
from autograph.lib.tree.sigma_tree_operation import SigmaTreeOperation
from autograph.lib.tree.tree_operation import TreeOperation
from autograph.lib.util import StatsTracker

# added by brett
from autograph.lib.shaping import AutShapingWrapperAdv
T = TypeVar("T")
INTRINS_CONSTANT = 10

BackupElement = Tuple[T, int, List[int], T, Union[Set[Tuple[int, int]], List[FrozenSet[Tuple[int, int]]]]]
BackupTrace = List[BackupElement]


class MultIndexArray:
    def __init__(self, arr, num_items):
        self.arr = arr
        self.num_items = num_items

    def __getitem__(self, item):
        if isinstance(item, slice):
            return MultIndexArray(self.arr[item], self.num_items)
        else:
            from_state, to_state = item
            return self.arr[from_state * self.num_items + to_state]

    def __setitem__(self, key, value):
        from_state, to_state = key
        self.arr[from_state * self.num_items + to_state] = value

    def __iter__(self):
        return self.arr.__iter__()

    def __repr__(self):
        return "MultiIndexArray(" + repr(self.arr) + ")"

    def indices(self):
        return np.ndindex((len(self.arr) // self.num_items, self.num_items))

    def copy(self):
        return MultIndexArray(self.arr.copy(), self.num_items)


class AbstractAutStats(metaclass=ABCMeta):
    @abstractmethod
    def baseline(self):
        pass

    @abstractmethod
    def v(self, state):
        pass

    @abstractmethod
    def synchronize(self):
        pass

    @abstractmethod
    def indices(self):
        pass

    def set_step(self, step):
        pass


class AutStats(AbstractAutStats):
    def __init__(self, num_items, uct_numerator: Union[int, None] = None):
        self.num_items = num_items
        self.n = MultIndexArray(Array("i", num_items ** 2, lock=False), num_items)
        self.w = MultIndexArray(Array("d", num_items ** 2, lock=False), num_items)
        self.arr_lock = Lock()
        self.local_n = self.n[:]
        self.local_w = self.w[:]
        self.max = 0
        self.base = 0
        self.uct_numerator = uct_numerator

    def baseline(self):
        return self.base

    def transition_index(self, from_state, to_state):
        return from_state * self.num_items + to_state

    def indices(self):
        return self.local_n.indices()

    def v(self, state):
        if isinstance(state, (frozenset, set)):
            if len(state) > 0:
                state = set(state).pop()
            else:
                return 0

        if self.local_w[state] > 0 and self.local_n[state] > 0 and self.max > 0:
            val = self.local_w[state] / (self.local_n[state] * self.max)
        else:
            val = 0

        if self.uct_numerator:
            val += (self.uct_numerator / (self.local_n[state] + self.uct_numerator))

        return val

    def synchronize(self):
        with self.arr_lock:
            self.local_n = self.n[:]
            self.local_w = self.w[:]
        self.max = max([w / n if n > 0 else 0 for n, w in zip(self.local_n, self.local_w)])
        # self.base = sum(self.local_w) / (sum(self.local_n) * self.max) if self.max > 0 else 0
        self.base = min(
            self.v(state) for state in self.indices() if self.local_n[state] > 0) if self.max > 0 else (
            1 if self.uct_numerator else 0)

    def visit(self, state, final_value):
        if isinstance(state, (frozenset, set)):
            if len(state) == 0:
                return
            state = set(state).pop()

        with self.arr_lock:
            self.n[state] += 1
            self.local_n = self.n[:]
            self.w[state] += final_value
            self.local_w = self.w[:]

    def state_dict(self):
        self.synchronize()
        return {
            "n": self.local_n.copy(),
            "w": self.local_w.copy()
        }

    def load_state_dict(self, sd):
        with self.arr_lock:
            n, w = sd["n"], sd["w"]
            for i in self.indices():
                self.n[i] = n[i]
                self.w[i] = w[i]

        self.synchronize()


class UCBAnnealedAutStats(AbstractAutStats):
    def __init__(self, anneal_from: AutStats, anneal_to: AutStats, rate: float):
        """
        Note: the higher the rate (1 <= rate < inf), the slower this anneals.
        """
        self.anneal_from = anneal_from
        self.anneal_to = anneal_to
        self.rate = rate
        self.base = 0

    def indices(self):
        return self.anneal_to.indices()

    def synchronize(self):
        self.anneal_to.synchronize()
        self.anneal_from.synchronize()

        self.base = min(self.v(state) for state in self.anneal_to.indices() if self.anneal_to.local_n[state] > 0) \
            if self.anneal_to.max > 0 else self.anneal_from.baseline()  # TODO generic with AbstractAutStats

    def v(self, state):
        if isinstance(state, (frozenset, set)):
            if len(state) == 0:
                return
            state = set(state).pop()

        from_v = self.anneal_from.v(state)
        to_v = self.anneal_to.v(state)
        anneal_step = self.anneal_to.local_n[state]
        ratio = self.rate / (anneal_step + self.rate)
        # Visit transition only a few times: ratio->1
        # Visit transition a lot: ratio->0
        return (ratio * from_v) + ((1 - ratio) * to_v)

    def baseline(self):
        return self.base


class ExponentialAnnealedAutStats(AbstractAutStats):
    def __init__(self, anneal_from: AbstractAutStats, anneal_to: AbstractAutStats, rate: float):
        self.anneal_from = anneal_from
        self.anneal_to = anneal_to
        self.rate = rate
        self.local_step = 0
        self.step = Value("i", 0)

    def set_step(self, step: int):
        self.step.value = step
        self.local_step = step

    def indices(self):
        return self.anneal_to.indices()

    def proportioned(self, old, new):
        old_proportion = self.rate ** self.local_step
        return (old_proportion * old) + ((1 - old_proportion) * new)

    def baseline(self):
        return self.proportioned(self.anneal_from.baseline(), self.anneal_to.baseline())

    def synchronize(self):
        self.local_step = self.step.value
        self.anneal_from.synchronize()
        self.anneal_to.synchronize()

    def v(self, state):
        return self.proportioned(self.anneal_from.v(state), self.anneal_to.v(state))


class MCTSAutAdvList(Generic[T]):
    def __init__(self, num_actions: int,
                 curiosity_evaluator: Callable[[List[BackupElement]], List[int]],
                 state_evaluator: Callable[[List[T]], List[Tuple[Sequence[float], float]]],
                 curiosity_trainer: Callable[[List[BackupElement]], None],
                 aut_stats: List[AutStats],
                 dont_use_aut: List[bool],
                 team_membership: List[int],
                 c_puct: float = 1.0, mcts_to_completion=False, discount=1, c_aut: float = 1.0, c_sigma: float = 1.0,
                 c_intrins: float = 1.0, c_intrins_add: float = 0, scale_aut: bool = False, aut_prob: bool = False):
        """
        :param c_puct: How much to weight the randomness and the NN-calculated probabilities as opposed to values
        :param mcts_to_completion: Keep traversing tree until end of episode, otherwise stop when we reach a leaf and
        use the value estimate as the reward function
        :param curiosity_trainer: A function that accepts a list of (state, action) pairs and uses this
        information to train the curiosity metric that these states have been visited
        :param state_evaluator: Given a list of states, give back a list of (action probabilities, value estimate)
        :param curiosity_evaluator: List of (state, action) pairs to list of raw curiosity values
        """

        self.scale_aut = scale_aut
        self.c_sigma = c_sigma
        self.mcts_to_completion = mcts_to_completion
        self.curiosity_evaluator = curiosity_evaluator
        self.state_evaluator = state_evaluator
        self.curiosity_trainer = curiosity_trainer
        self.num_actions = num_actions
        self.discount = discount
        self.aut_stats = aut_stats
        self.aut_prob = aut_prob
        self.team_membership = team_membership
        self.c_puct = c_puct
        self.c_aut = c_aut
        self.c_intrins = c_intrins
        self.c_intrins_add = c_intrins_add
        self.dont_use_aut = dont_use_aut

       # count of visits, state_int -> [N(s, a)]
        self.visit_count = {}
        # total value of the state's action, state_int -> [W(s, a)]
        self.value = {}
        # average value of actions, state_int -> [Q(s, a)]
        self.value_avg = {}
        # prior probability of actions, state_int -> [P(s,a)]
        #self.probs = list({} for i in range(0, len(aut_stats)))
        self.probs = {}
        # most recently computed intrinsic reward for single action step, state -> [IR(s, a)]
        self.intrinsic = {}

        self.hideAndSeek = True
        #shared between agents
        # tree uncertainty https://arxiv.org/pdf/1805.09218.pdf
        self.sigma = {}

        self.intrinsic_maxchildren = {}

        self.intrinsic_stats = StatsTracker()

        self.state_action_aut_total = {}
        self.state_action_aut_average = {}

        # added nested list of floats (list of probs) to deal with backwards passing multiple values -bb
        # The properties to backup during the backwards MCTS pass, along with how to actually do the backup
        self.tree_backups: List[Tuple[Dict[Any, List[List[float]]], TreeOperation]] = [
            (self.intrinsic, MaxTreeOperation()),
            (self.sigma, SigmaTreeOperation()),
            (self.state_action_aut_average, MaxTreeOperation())
        ]

    def clear(self) -> None:
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.probs.clear()
        self.sigma.clear()
        self.intrinsic.clear()
        self.intrinsic_maxchildren.clear()
        self.state_action_aut_total.clear()
        self.state_action_aut_average.clear()

    def __len__(self) -> int:
        return len(self.value)


    def render_tree_step(self, treestep, env, batchnum, leaf_num):
        print("tree step " + str(treestep) + "leaf_" + str(leaf_num) + "BATCH_" + str(batchnum))
        env.render()


    def pick_action(self, state: T, root_state: bool) -> int:
        # this is \pi_tree eqn implementation (7)


        num_teams = len(self.aut_stats)

        # Get the current team
        # this currently only works if we have an odd and even team listed in alternating order
        # could also work for 3+ teams, but agent init_positions would need to be in alternating order
        # so play will alternate teams each turn, we access values from each team
        #cur_turn = state[3] % num_teams
        cur_turn = self.team_membership[state[3]]

        #same for both teams
        counts = self.visit_count[state]
        total_count_sqrt = max(math.sqrt(sum(counts)), 1)

        #diff for each team
        #TODO: check this
        probs = self.probs[state][(cur_turn)]

        values_avg = self.value_avg[state]

        # is curiosity / instinsic being used??
        intrins = self.intrinsic_maxchildren[state]
        intrins = [i if i is not None else self.intrinsic_stats.average() for i in intrins]

        total_intrins = sum(intrins)
        if total_intrins == 0:
            intrins_normalized = np.zeros_like(intrins)
        else:
            intrins_normalized = [i / total_intrins for i in intrins]

        aut_avg = self.state_action_aut_average[state]
        aut_avg = [self.aut_stats[cur_turn].baseline() if a[cur_turn] is None else a[cur_turn] for a in aut_avg]

        if self.scale_aut:
            aut_avg_min = min(aut_avg)
            aut_range = 1 - aut_avg_min
            if aut_range > 0:
                aut_avg = [(a - aut_avg_min) / aut_range for a in aut_avg]

        #agents share use of sigma
        sigma = self.sigma[state]

        if root_state:
            # Add random noise to (NN-calculated) probabilities
            noises = np.random.dirichlet([0.03] * self.num_actions)
            probs = [0.75 * prob + 0.25 * noise for prob, noise in zip(probs, noises)]


        # we could also do this by using an array for c_aut values and referencing
        # depending on turn. self.c_aut[turn] * (aut v.......**below)
        if self.dont_use_aut[cur_turn]:
            aut_avg = [None] * len(aut_avg)

        # Calculate score based on
        # TODO: get specific c_aut based on turn?
        score = [
            value[cur_turn]
            + self.c_aut * (aut_v if aut_v is not None else 0)
            * (prob if self.aut_prob else 1)
            + self.c_intrins_add * intrin
            + self.c_puct * prob * (total_count_sqrt / (1 + count))
            * ((1 - self.c_sigma) + (sigma * self.c_sigma))
            * ((1 - self.c_intrins) + (intrin * self.c_intrins))
            for value, prob, intrin, count, sigma, aut_v
            in zip(values_avg, probs, intrins_normalized, counts, sigma, aut_avg)]

        action = int(np.argmax(score))

        Q = [values_avg[action][cur_turn] for action in range(self.num_actions)]
        Y = aut_avg
        pi_cnn = probs

        return action, score, Q, Y, pi_cnn

    def find_leaf(self, envs: List[AutShapingWrapperAdv], cur_state: T, aut_states: List[Set[int]],
                  batch_num: int = 0, leaf_num: int = 0, #for rendering purposes and debugging
                  show_tree_step: bool = False) -> Tuple[BackupTrace, T, bool]:

        # THIS IS THE MAIN VGTS FUNC

        """
        Starting from the root state, traverse MCTS tree until either a leaf is found or a terminal state is reached
        :param envs: The environments/autwrappers to run the simulation in
        :param cur_state: The root state
        :return: Trajectory taken, last state, is done
        """

        turns = []  # Tuple of state, action, reward, next state: this is for everyone.
                    #alternates POV per turn (POV is merged by team, not individual).
        seenstates = {cur_state}

        done = False
        root = True
        treestep = 0
        while not done and not self.is_leaf(cur_state):
            prev_aut_states = aut_states

            #team turn is decided by modulo number of autonama
            # players must be listed in json as alternating teams
            team_turn = cur_state[3] % len(self.aut_stats)

            action, _, _, _, _ = self.pick_action(cur_state, root)

            root = False
            rewards = [0] * len(self.aut_stats)
            dones = [False] * len(self.aut_stats)
            info = [None] * len(self.aut_stats)
            aut_states = [frozenset()] * len(self.aut_stats)
            next_states = [None] * len(self.aut_stats)
            # gather steps once per team
            for i in range(0, len(self.aut_stats)):
                next_states[i], rewards[i], dones[i], info[i] = envs[i].step(action)

                # NOTE: if we have partial_obs false in our environments,
                #       this will be a full state.
                #       if partial_obs is true in our json, we get partial next


                # SHOW TREE STEP
                if show_tree_step and i == 0:
                    self.render_tree_step(treestep, envs[0], batch_num, leaf_num)
                    treestep += 1


                aut_states[i] = frozenset(info[i]["automaton_states"])

                # if one team wins (done from their env) give all other teams their terminate on fail reward
            for idx, dun in enumerate(dones):
                if dun:
                    done = True
                    for jdx in range(0, len(rewards)):
                        if jdx != idx:
                            rewards[jdx] += envs[jdx].termination_fail_reward


            aut_edges = [frozenset()] * len(self.aut_stats)
            for i in range(0, len(self.aut_stats)):
                if len(aut_states[i]) == 0 or len(prev_aut_states[i]) == 0:
                    aut_edges[i] = (frozenset())
                else:
                    aut_edges[i] = (frozenset({(set(prev_aut_states[i]).pop(),
                                                set(aut_states[i]).pop())}))


            next_state_next_team = next_states[(team_turn + 1) % len(self.aut_stats)]
            turns.append((cur_state, action, tuple(rewards),
                          next_state_next_team,
                          tuple(aut_edges)))

            if self.c_sigma != 0:
                if done or next_state_next_team in seenstates:
                    done = True
                    self.sigma[cur_state][action] = 0
                else:
                    seenstates.add(next_state_next_team)
            cur_state = next_state_next_team

            for d in dones:
                if d == True:
                    done = True

        return turns, cur_state, done

    def find_leaf_batch(self, envs: List[AutShapingWrapperAdv], num_leaves: int, start_state: T, num_batch: int) \
            -> Tuple[List[BackupElement], List[BackupTrace], List[Tuple[BackupTrace, T]]]:
        """
        Run multiple MCTS simulations and aggregate some of the results
        :param envs: Copyable environments
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

        env_states = list(env.save_state() for env in envs)

        for i in range(num_leaves):
            turns, end_state, done = self.find_leaf(envs, start_state,
                                                    list(env.current_automaton.states for env in envs),
                                                    batch_num=num_batch, leaf_num=i)

            intrinsic_to_calculate.update(turns)

            if done:
                backup_queue.append(turns)
            else:
                if end_state not in will_create:
                    will_create.add(end_state)
                    create_queue.append((turns, end_state))

            # Reset environment / AUT wrappers
            for i in range(0, len(self.aut_stats)):
                envs[i].load_state(env_states[i])

        return list(intrinsic_to_calculate), backup_queue, create_queue


#TODO curiousity for multiagent
    def update_curiosity(self, states: List[BackupElement]) -> None:
        """
        Use the curiosity evaluator to update intrinsic rewards
        :param states: The states to update the intrinsic rewards in
        """
        curiosity_values = self.curiosity_evaluator(states)

        for i, (state, action, reward, next_state, aut) in enumerate(states):
            if self.intrinsic[state][action] is not None:
                self.intrinsic_stats.remove_point(self.intrinsic[state][action])
            self.intrinsic[state][action] = curiosity_values[i]
            self.intrinsic_maxchildren[state][action] = curiosity_values[i]
            self.intrinsic_stats.add_point(curiosity_values[i])

    # grab turn from the cur_state and add to self counters like a double indexed array
    def create_state(self, cur_state: T, probs: List[Sequence[float]]) -> None:
        """
        Create an empty state
        :param cur_state: The state to create
        :param probs: The action probabilities for the state
        """
        num_actions = self.num_actions
        num_teams = len(probs)

        self.visit_count[cur_state] = [0] * num_actions
        vals = [0.0] * num_teams
        self.value[cur_state] = [vals[:] for i in range(0, num_actions)]
        self.value_avg[cur_state] = [vals[:] for i in range(0, num_actions)]

        # this is also a multi indexed probs now because of input
        self.probs[cur_state] = probs.copy()
        self.intrinsic[cur_state] = [None] * num_actions
        self.intrinsic_maxchildren[cur_state] = [None] * num_actions
        self.sigma[cur_state] = [1.0] * num_actions

        self.state_action_aut_total[cur_state] = [vals[:] for i in range(0, num_actions)]
        self.state_action_aut_average[cur_state] = [vals[:] for i in range(0, num_actions)]

    def mcts_mini_batch(self, envs: List[AutShapingWrapperAdv], start_state: T, batch_size: int, num_batch: int) -> None:
        """
        Run a minibatch of mcts and update curiosity metrics
        :param env: Environment to run the minibatch in. Must be copyable
        :param start_state: Root state to start search from
        :param batch_size: How many MCTS simulations to take
        """
        intrinsic_calculate, backups, creates = self.find_leaf_batch(envs,
                                                                     start_state=start_state,
                                                                     num_leaves=batch_size,
                                                                     num_batch=num_batch)
        if len(creates) > 0:
            create_trajectories, create_states = zip(*creates)

            #evaluates state on both nets
            state_infos = list(self.state_evaluator(create_states, i)
                               for i in range(0, len(self.aut_stats)))

        else:
            create_trajectories, create_states = [], []
            state_infos = list(create_states[:] for i in range(0, len(self.aut_stats)))


        backups_with_final_values = []
        for bup in backups:
            backups_with_final_values.append((bup, [0] * len(self.aut_stats)))

        policies = list(list(state_infos[agent][i][0] for agent in range(0, len(self.aut_stats)))
                        for i in range(0, len(create_states)))

        values = list(list(state_infos[agent][i][1] for agent in range(0, len(self.aut_stats)))
                        for i in range(0, len(create_states)))
        # what to do here?
        idx = 0
        for state, bup in zip(create_states, create_trajectories):

            self.create_state(state, policies[idx]) # we only need to create one state here

            #send both values to the backup
            backups_with_final_values.append((bup, values[idx]))
            idx += 1
        if len(intrinsic_calculate) > 0 and (self.c_intrins > 0 or self.c_intrins_add > 0):
            self.update_curiosity(intrinsic_calculate)
            self.curiosity_trainer(intrinsic_calculate)

        for bup, values in backups_with_final_values:
            self.backup_mcts_trace(bup, values)

    def is_leaf(self, state: T) -> bool:
        """
        Is the given state not expanded yet?
        """
        return state not in self.probs

    def backup_mcts_trace(self, sars: BackupTrace, final_value_estimates: List[float]) -> List[float]:
        """
        Given a trace of a MCTS forward run, use the final value estimates and the actual rewards to backup the
        tree node values
        :param sars: List of tuples of state, action, reward
        :param final_value_estimates: A final value estimate for both agents if the MCTS doesn't reach the terminal state.
        If the MCTS runs a simulation until termination, this should be 0.
        :return The value of the first state
        """
        values_discounted = final_value_estimates

        prev_tree = [None] * len(self.aut_stats)

        prev_tree_values = [prev_tree[:] if i == len(self.tree_backups)-1 else None for i in range(0, len(self.tree_backups))]

        for state, action, rewards, next_state, aut_states in reversed(sars):

            self.visit_count[state][action] += 1

            # calculate for each team. adjusted values in set indexed by state->action->team
            for team in range(0, len(self.aut_stats)):
                self.state_action_aut_total[state][action][team] += self.aut_stats[team].v(aut_states[team])
                self.state_action_aut_average[state][action][team] = \
                    self.state_action_aut_total[state][action][team] / self.visit_count[state][action]

                # Backup values according to the tree operations defined in the constructor
            for idx, ((storage, treeop), prev_values) in enumerate(zip(self.tree_backups, prev_tree_values)):
                # values backup happens during second index.
                if idx == 2 and prev_values[0] is not None:
                    for team in range(0, len(self.aut_stats)):
                        storage[state][action][team] = \
                            treeop.edge_combinator(storage[state][action][team], prev_values[team])
                elif prev_values is not None and type(prev_values) is float:
                    # for our intrinsic and sigmas backup
                    storage[state][action] = \
                        treeop.edge_combinator(storage[state][action], prev_values)
                if idx == 2:
                    # changes the indexing of our storage so different action values accessible by team index
                    storage_split = list(
                        list(storage[state][action][team] for action in range(0, self.num_actions))
                        for team in range(0, len(self.aut_stats)))

                    # grab storage backups and put into prev_tree_values by team
                    for team in range(0, len(self.aut_stats)):
                        prev_tree_values[idx][team] = treeop.node_value(storage_split[team])
                else:
                    # for intrinsic and sigma passing
                    prev_tree_values[idx] = treeop.node_value(storage[state])

            # reward is 1 if we found an accepting node in the automaton
            # reward is only 1 at the last node of the trajectory, and 0 for all other nodes
            for idx, value_discounted in enumerate(values_discounted):
                value_discounted *= self.discount
                value_discounted += rewards[idx]
                #redundant?
                values_discounted[idx] = value_discounted

            for i in range(0, len(self.aut_stats)):

                # value_avg is Q
                self.value[state][action][i] = self.value[state][action][i] + values_discounted[i]
                self.value_avg[state][action][i] = self.value[state][action][i] / self.visit_count[state][action]

        return values_discounted

    def mcts_batch(self, envs: List[AutShapingWrapperAdv], state: Any, count: int, batch_size: int) -> None:
        """
        Run a batch of MCTS
        :param env: The environment to run the batch in. Must be copyable
        :param state: The state we are currently in
        :param count: How many minibatches to run
        :param batch_size: Size of each minibatch
        """
        for num_batch in range(count):
            self.mcts_mini_batch(envs, state, batch_size, num_batch)

    # returns pi_play
    def get_policy_value(self, state: T, tau: float = 1) -> Tuple[List[float], List[List[float]]]:
        """
        Extract policy and action-values by the state
        :param state: state of the board
        :param tau: temperature (as defined in alphago zero paper)
        :return: (probs, values)
        """
        turn = state[3]
        if True:
            #should be able to hold states for both agents because turn is included
            counts = self.visit_count[state]
            if tau == 0:  # Set the best action to 1, others at 0
                probs = [0.0] * len(counts)
                probs[np.argmax(counts).item()] = 1.0
            else:  # Make the best action stand out more or less
                counts = [count ** (1.0 / tau) for count in counts]
                total = sum(counts)
                probs = [count / total for count in counts]

        # get values for both teams
        values = list(list(self.value_avg[state][action][team] for action in range(0, self.num_actions))
                      for team in range(0, len(self.aut_stats)))
        #get policy for player that is playing
        return probs, values







