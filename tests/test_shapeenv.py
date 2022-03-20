from unittest import TestCase

from gym import Env

from autograph.lib.automata import AutomatonSet
from autograph.lib.hoa import parse
from autograph.lib.shaping import AutShapingWrapper

aut = """
HOA: v1
name: "G(!positive | Fnegative)"
States: 2
Start: 0
AP: 2 "positive" "negative"
acc-name: Buchi
Acceptance: 1 Inf(0)
properties: trans-labels explicit-labels state-acc complete
properties: deterministic stutter-invariant
--BODY--
State: 0 {0}
[!0 | 1] 0
[0&!1] 1
State: 1
[1] 0
[!1] 1
--END--"""


class ToyEnv(Env):
    def __init__(self):
        self.sum = None

    def reset(self):
        self.sum = 0
        return 0

    def step(self, action):
        self.sum += action
        reward = 1 if self.sum % 2 == 0 else 0
        return self.sum, reward, False, dict()


def positive(action, obs, rew, done, info):
    return obs > 0


def negative(action, obs, rew, done, info):
    return obs < 0


aut = AutomatonSet.from_hoa(parse(aut))


class ShapeEnvTest(TestCase):
    def test_reward_shaping(self):
        env = ToyEnv()
        env = AutShapingWrapper(env, [positive, negative], aut, 1, 0.9, use_potential=True)

        def st(action):  # Helper to extract reward from result of step
            _, rew, _, _ = env.step(action)
            return rew

        env.reset()

        self.assertEqual(.9, st(0))  # Even, no state transitions, but acceptance is discounted
        self.assertEqual(-1, st(1))  # Odd, transition away from acceptance
        self.assertEqual(0, st(2))  # Odd, no transition
        self.assertEqual(1.9, st(-5))  # Even, transition to acceptance
