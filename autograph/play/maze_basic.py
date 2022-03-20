from statistics import mean

from autograph.lib.automata import AutomatonSet
from autograph.lib.envs.mazeenv import FuelMazeEnv
from autograph.lib.hoa import parse
from autograph.lib.mcts import MCTS
from autograph.lib.shaping import AutShapingWrapper

print_every = 1000

fuel_hoa_text = """
HOA: v1
name: "GF!low"
States: 2
Start: 0
AP: 1 "low"
acc-name: Buchi
Acceptance: 1 Inf(0)
properties: trans-labels explicit-labels state-acc complete
properties: deterministic stutter-invariant
--BODY--
State: 0
[0] 0
[!0] 1
State: 1 {0}
[0] 0
[!0] 1
--END--"""

fuel_hoa = parse(fuel_hoa_text)
fuel_aut = AutomatonSet.from_hoa(fuel_hoa)

key_hoa_text = """
HOA: v1
name: "Fkey"
States: 2
Start: 1
AP: 1 "key"
acc-name: Buchi
Acceptance: 1 Inf(0)
properties: trans-labels explicit-labels state-acc complete
properties: deterministic stutter-invariant terminal
--BODY--
State: 0 {0}
[t] 0
State: 1
[0] 0
[!0] 1
--END--"""

key_hoa = parse(key_hoa_text)
key_aut = AutomatonSet.from_hoa(key_hoa)

discount = .9
keys = 1

USE_AUT = True


def has_key(n):
    def has_key_n(action, obs, rew, done, info):
        return obs[1][n] == 0

    return has_key_n


def low_fuel(action, obs, rew, don, info):
    return obs[2] < 11


# 234- works nicely
# 12345- reach key just as we run out of fuel

def run():
    env = FuelMazeEnv(shape=(6, 6), num_keys=keys, loop_factor=.0, max_fuel_dist=10, max_fuel_level=20, seed=12345)

    if USE_AUT:
        env = AutShapingWrapper(env, [low_fuel], fuel_aut, 1, discount, use_potential=True)
        for i in range(keys):
            env = AutShapingWrapper(env, [has_key(i)], key_aut, 1, discount, use_potential=True)

    mcts = MCTS(mcts_to_completion=False)
    action_space = env.action_space.n
    pr_num = 0
    recent_rewards = []
    recent_lengths = []

    def evaluator(state):
        return [1 / action_space] * action_space, 0

    while True:
        trace, val_estimate = mcts.mcts_round(env, evaluator, discount)
        recent_rewards.append(sum((reward for _, _, reward in trace)))
        recent_lengths.append(len(trace))

        pr_num += 1
        if pr_num % print_every == 0:
            print("{:}-avg:{:.3f}, min:{:.3f}, max:{:.3f}, avgl:{:.3f}"
                  .format(pr_num, mean(recent_rewards), min(recent_rewards), max(recent_rewards), mean(recent_lengths)))

            recent_rewards.clear()
            recent_lengths.clear()

            env.render()


if __name__ == '__main__':
    run()
