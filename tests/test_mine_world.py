from collections import Counter
from unittest import TestCase
import unittest

from autograph.lib.envs.mineworldenv import MineWorldConfig, MineWorldEnv, MineWorldTileType, \
    InventoryItemConfig, TilePlacement


class TestMineWorld(TestCase):
    def test_empty_world(self):
        cfg = MineWorldConfig((10, 10), (0, 0), [], [])
        world = MineWorldEnv(cfg)

        obs, *_ = world.reset()
        self.assertEqual((0, 0), obs)

        for attempt in (0, 3, 4, 5):
            (obs, *_), *_ = world.step(attempt)
            self.assertEqual(0, 0), obs

        for i in range(1, 9):
            (obs, *_), *_ = world.step(1)
            self.assertEqual((i, i - 1), obs)
            (obs, *_), *_ = world.step(2)
            self.assertEqual((i, i), obs)

        for attempt in (1, 2, 4, 5):
            (obs, *_), *_ = world.step(attempt)
            self.assertEqual(0, 0), obs

    def test_basic_ap(self):
        cfg = MineWorldConfig((10, 10), (2, 4),
                              [TilePlacement(tile=MineWorldTileType(False, Counter(a=2, b=-1, c=1), "ap", "F"),
                                             fixed_placements=[(2, 4)])],
                              [InventoryItemConfig("a", 2, 5), InventoryItemConfig("b", 2, 2),
                               InventoryItemConfig("c", 0, 5)])
        world = MineWorldEnv(cfg)

        rounds = (
            (.4, 1, 0),
            (.8, .5, .2),
            (1, 0, .4),
            (1, 0, .4),
            (1, 0, .4)
        )
        obs = world.reset()

        for round in rounds:
            expected = ((2, 4), (frozenset({(2, 4)}),), round)
            self.assertEqual(expected, obs)

            obs, *_ = world.step(5)

    def test_ap_generation(self):
        cfg = MineWorldConfig((10, 10), (0, 0),
                              [TilePlacement(tile=MineWorldTileType(False, Counter(), "a", "F"),
                                             fixed_placements=[(4, 5)]),
                               TilePlacement(tile=MineWorldTileType(False, Counter(), "b", "F"), random_placements=98)],
                              [])

        world = MineWorldEnv(cfg)
        pos, (a_locs, b_locs), _ = world.reset()

        self.assertEqual(a_locs, frozenset({(4, 5)}))
        self.assertEqual(len(b_locs), 98)
        for loc in b_locs:
            self.assertNotEqual(loc, (4, 5))

unittest.main()