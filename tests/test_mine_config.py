#from collections import Counter
#from unittest import TestCase
import unittest

from autograph.lib.envs.mineworldenv_adv import MineWorldConfig, MineWorldEnv, MineWorldTileType, \
    InventoryItemConfig, TilePlacement


#class TestMineWorld(TestCase):
#def test_empty_world(self):
cfg = MineWorldConfig((10, 10), (0, 0), (2,2), [],[])
world = MineWorldEnv(cfg)

#obs, *_ = world.reset()
obs = world.reset()
        #self.assertEqual((0, 0), obs)
print(obs[4])

for attempt in (0, 3, 4, 5):
     obs, *_ = world.step(attempt)
     print('trun = ',obs[4])
     #world.render()