import unittest

from autograph.play.maze_nn_aut_adv_team_share import init_game_envsonly
from autograph.lib.util.test_helper_team import doenvsteps

class TestAutAgvTeamShare_seeBothInfs(unittest.TestCase):
    def setUp(self):
        config_file = '../autograph/play/config/pre_guardInf/seeBothInfs.json'
        self.envs = init_game_envsonly(config_file)

    def test_player0_win(self):
        # Sequence of actions for each player.
        # If a player's list is empty, then noops will be used for that player
        # If a player's list is shorter than the rest, noops will be used once the player's sequence ends
        actionlist = [[0,0,2,0,0,3,1,1,0,0,2,0,3,0,3], # player 0
                      [0,0],                           # player 1
                      [],                              # player 2
                      []]                              # player 3

        stepnum, done, lastplayer, lastrew, lastautstate, finishedrun = doenvsteps(self.envs, actionlist)
        self.assertTrue(done)
        self.assertEqual(lastplayer, 0)
        self.assertEqual(lastrew, 1)
        self.assertSetEqual(lastautstate, set([1]))

if __name__ == '__main__':
    unittest.main()