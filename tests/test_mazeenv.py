from unittest import TestCase, skip

import networkx as nx
import numpy as np

from autograph.lib.envs.mazeenv import FuelMazeEnv


def avg_degree(graph: nx.Graph):
    return (len(graph.edges) * 2) / len(graph.nodes)  # count each edge twice, one for each node it is connected to


def maze_env_props(env: FuelMazeEnv):
    return env.maze.nodes, env.maze.edges, env.keys, env.fuel


class TestMazeEnv(TestCase):
    def test_rand_seed_eq(self):
        maze = FuelMazeEnv(seed=234)
        maze2 = FuelMazeEnv(seed=234)

        self.assertEqual(maze_env_props(maze), maze_env_props(maze2))

        maze3 = FuelMazeEnv(seed=235)

        self.assertNotEqual(maze_env_props(maze), maze_env_props(maze3))

    def test_graph_sizes(self):
        maze = FuelMazeEnv(shape=(24, 13))
        self.assertEqual(24 * 13, len(maze.maze.nodes))

    def test_graph_tree(self):
        maze = FuelMazeEnv(loop_factor=0)
        self.assertTrue(nx.is_tree(maze.maze))

    def test_graph_not_tree(self):
        maze = FuelMazeEnv(loop_factor=0.1)
        self.assertFalse(nx.is_tree(maze.maze))

    def test_graph_connections(self):
        maze = FuelMazeEnv(shape=(50, 50), max_fuel_dist=200, loop_factor=0)
        self.assertAlmostEqual(2, avg_degree(maze.maze), delta=.1)

        maze2 = FuelMazeEnv(shape=(50, 50), max_fuel_dist=200, loop_factor=1)
        self.assertAlmostEqual(4, avg_degree(maze2.maze), delta=.1)

        maze3 = FuelMazeEnv(shape=(50, 50), max_fuel_dist=200, loop_factor=.5)
        self.assertAlmostEqual(3, avg_degree(maze3.maze), delta=.1)

    @skip("Observation changed considerably")
    def test_steps(self):
        maze = FuelMazeEnv(shape=(2, 2), num_keys=2, loop_factor=1, max_fuel_level=5)
        self.assertEqual({(0, 1), (1, 0)}, set(maze.keys))  # Only possibility

        maze.fuel = []  # Disable refueling (evil laughter)

        expect_pos = (0, 0)
        expect_key = [1, 1]
        expect_fuel = 5
        expect_done = False
        expect_reward = 0

        obs = maze.reset()
        reward = 0
        done = False

        def verify():
            self.assertTrue(np.array_equal(expect_pos, obs[0]))
            self.assertTrue(np.array_equal(expect_key, obs[1]))
            self.assertEqual(expect_fuel, obs[2])
            self.assertEqual(expect_done, done)
            self.assertAlmostEqual(expect_reward, reward, delta=0.00001)

        verify()

        obs, reward, done, _ = maze.step(0)  # Go up against the wall
        expect_fuel -= 1
        verify()

        obs, reward, done, _ = maze.step(1)  # Go right
        expect_fuel -= 1
        expect_pos = 1, 0
        if maze.keys[0] == (1, 0):  # If we are on first key
            expect_key = [0, 1]
        else:
            expect_key = [1, 0]

        verify()

        obs, reward, done, _ = maze.step(2)  # Go down
        expect_fuel -= 1
        expect_pos = 1, 1
        verify()

        obs, reward, done, _ = maze.step(3)  # Left
        expect_fuel -= 1
        expect_pos = 1, 0
        expect_key = [0, 0]

        obs, reward, done, _ = maze.step(1)
        expect_fuel = 0
        expect_pos = 1, 1
        expect_done = True
        expect_reward = 1000

    @skip("Observation changed considerably")
    def test_steps_out_of_fuel(self):
        maze = FuelMazeEnv(shape=(2, 2), loop_factor=1, max_fuel_level=11)
        maze.fuel = [(1, 0)]

        exfuel = 11
        maze.reset()

        def run_down(exfuel, num):
            for i in range(num):
                if i % 2 == 0:
                    obs, rew, done, _ = maze.step(2)
                else:
                    obs, rew, done, _ = maze.step(0)
                exfuel -= 1

                self.assertEqual(exfuel, obs[2])
                self.assertEqual(False, done)

        run_down(exfuel, 10)

        obs, rew, done, _ = maze.step(1)  # Get that fuel
        exfuel = 11
        self.assertEqual(exfuel, obs[2])
        self.assertFalse(done)

        obs, rew, done, _ = maze.step(3)
        exfuel -= 1
        self.assertEqual(exfuel, obs[2])

        run_down(exfuel, 9)

        obs, rew, done, _ = maze.step(0)
        self.assertEqual(0, obs[2])
        self.assertEqual(0, rew)
        self.assertEqual(True, done)
