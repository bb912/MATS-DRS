from collections import Counter
from random import randrange
from unittest import TestCase

import numpy as np
import numpy.testing as npt

from autograph.lib.envs.adversarialenv import PlayerID
from autograph.lib.envs.connfourenv import ConnectFourEnv


class TestBoard(TestCase):
    def setUp(self):
        self.board = ConnectFourEnv()
        self.board.reset()

    def test_players(self):
        board = self.board

        self.assertEqual(board.get_current_player(), PlayerID.Player1)

        board.step((PlayerID.Player1, 0))
        self.assertEqual(board.get_current_player(), -1)

        board.step((PlayerID.Player2, 0))
        self.assertEqual(board.get_current_player(), 1)

    # noinspection PyTypeChecker
    # Intentionally badly typed code here
    def test_bad_players(self):
        board = self.board

        self.assertRaises(Exception, lambda: board.step((PlayerID.Player2, 0)))
        self.assertEqual(np.sum(board.board), 0)  # All spaces are empty still
        self.assertEqual(board.get_current_player(), 1)

        board.step((PlayerID.Player1, 0))
        self.assertEqual(np.sum(board.board), PlayerID.Player1)  # Player 1 moved

        self.assertRaises(Exception, lambda: board.step((PlayerID.Player1, 0)))
        self.assertEqual(np.sum(board.board), 1)  # There isn't a player -1 yet
        self.assertEqual(board.get_current_player(), PlayerID.Player2)

        self.assertRaises(Exception, lambda: board.step((0, 0)))
        self.assertEqual(np.sum(board.board), 1)  # There still isn't a player -1 yet
        self.assertEqual(board.get_current_player(), PlayerID.Player2)

    def test_board_move(self):
        board = self.board

        for column in range(7):
            obs = board.reset()
            npt.assert_equal(obs, np.zeros(shape=(7, 6), dtype=int))

            player = board.get_current_player()
            npt.assert_equal(player, PlayerID.Player1)

            for row in range(6):
                obs, reward, done, _ = board.step((player, column))

                self.assertEqual(reward, 0)
                self.assertEqual(done, False)

                for testcol in range(7):
                    for testrow in range(6):
                        if testcol == column and testrow <= row:  # The piece is already there
                            self.assertEqual(obs[testcol][testrow],
                                             PlayerID.Player1 if testrow % 2 == 0 else PlayerID.Player2)
                        else:
                            self.assertEqual(obs[testcol][testrow], 0)

                player *= -1

    def test_bad_moves(self):
        board = self.board

        const_moves = tuple(((col,) * 6, col) for col in range(7))

        bad_moves = (
            ((), -1),
            ((), 7),
            ((2, 5, 5, 2), -1),
            *const_moves
        )

        for trajectory, final_move in bad_moves:
            obs = board.reset()
            player = board.get_current_player()

            for move in trajectory:
                obs, _, _, _ = board.step((player, move))
                player *= -1

            board_before = obs.copy()
            self.assertRaises(Exception, lambda: board.step((player, final_move)))
            npt.assert_equal(board_before, board.board)

    def test_allowed_moves(self):
        board = self.board

        for i in range(7):
            board.reset()
            player = board.get_current_player()

            expected = np.full(shape=(7,), fill_value=True)

            for move in range(6):
                npt.assert_equal(board.get_allowed_moves(), expected)
                board.step((player, i))
                player = -player

            expected[i] = False
            npt.assert_equal(board.get_allowed_moves(), expected)



    def test_game_over(self):
        board = self.board

        game_overs = (
            ((3, 3, 4, 4, 6, 6), 5),  # Horizontal (P1 and P2)
            ((0, 0, 1, 1, 2, 2, 4, 3, 4), 3),
            ((1, 2, 1, 2, 1, 2), 1),  # Vertical
            ((2, 3, 6, 3, 4, 3, 5), 3),
            ((1, 2, 2, 3, 3, 4, 3, 4, 4, 6), 4),  # Diagonal up right
            ((6, 1, 2, 2, 3, 3, 4, 3, 4, 4, 6), 4),
            ((5, 6, 4, 5, 4, 4, 3, 3, 3), 3),  # Diagonal up left
            ((0, 5, 6, 4, 5, 4, 4, 3, 3, 3), 3)
        )

        for trajectory, final_move in game_overs:
            obs = board.reset()
            player = board.get_current_player()

            heights = Counter()

            for move in trajectory:
                self.assertEqual(obs[move][heights[move]], 0)
                obs, reward, done, _ = board.step((player, move))
                self.assertEqual(obs[move][heights[move]], player)
                self.assertEqual(reward, 0)
                self.assertFalse(done)

                heights[move] += 1

                player *= -1

            obs, reward, done, _ = board.step((player, final_move))

            self.assertEqual(obs[final_move][heights[final_move]], player)
            self.assertEqual(reward, 1)
            self.assertTrue(done)

            # Don't allow moves after game is done
            self.assertRaises(Exception, lambda: board.step((-1 * player, randrange(7))))
