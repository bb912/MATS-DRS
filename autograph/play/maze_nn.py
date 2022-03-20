import math

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from autograph.lib.envs.mazeenv import FuelMazeEnv
from autograph.lib.envs.mazeenv import transform_coordinate
from autograph.lib.loss_functions import TakeSimilarActionsLossFunction
from autograph.lib.running import run_mcts_episode, \
    get_parallel_queue, RandomReplayTrainingLoop
from autograph.lib.util.checkpoint_manager import CheckpointManager, StateDictLoadHandler, CombinedLoadHandler, \
    InitZeroLoadHandler
from autograph.net import Mazenet
from autograph.net.curiosity.curiosity_optimizer import ModuleCuriosityOptimizer
from autograph.net.curiosity.rnd_models import RND

math.sqrt(1)  # So that the import isn't optimized away (very useful when setting conditional debug breakpoints)

DISCOUNT = 1

# Maze parameters
KEYS = 1
MAZE_SHAPE = (10, 10)
FUEL_CAP = 20
MAX_FUEL_DISTANCE = 10
LOOP_FACTOR = .3
RANDOM_SEED = 12345
RANDOMIZE_ON_RESET = True
MAX_EPISODE_LEN = 500

# Policy training hyperparameters
LEARNING_RATE = .001
CURIOSITY_LEARNING_RATE = .0001
REPLAY_BUFFER = 10000
MIN_TRACE_TO_TRAIN = 100
PPO_TRAIN_ROUNDS = 10

# Policy MCTS parameters
MCTS_NUM_BATCHES = 50
MCTS_BATCH_SIZE = 4
MCTS_C_PUCT = 1

# Curiosity Parameters
ICM_FEATURE_SPACE = 100

# Logging and checkpointing
LOG_FOLDER = "runs/mcts/testing4"
CHECKPOINT_PATH = "checkpoints/testing4"
SAVE_CHECKPOINTS = True
CHECKPOINT_EVERY = 1
LOAD_FROM_CHECKPOINT = True

NUM_PROCESSES = 8
CUDA = True
DEVICE = torch.device("cuda:0" if CUDA else "cpu")


def run():
    env = FuelMazeEnv(shape=MAZE_SHAPE, num_keys=KEYS, loop_factor=LOOP_FACTOR, max_fuel_level=FUEL_CAP,
                      max_fuel_dist=MAX_FUEL_DISTANCE, seed=RANDOM_SEED, randomize_on_reset=RANDOMIZE_ON_RESET)

    action_space = env.action_space.n
    writer = SummaryWriter(LOG_FOLDER)

    cman = CheckpointManager(CHECKPOINT_PATH, LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS)

    net = cman.load("net", Mazenet(transform_coordinate(MAZE_SHAPE), action_space), StateDictLoadHandler()).to(DEVICE)
    net.share_memory()

    icm = cman.load("icm", RND(7, transform_coordinate(MAZE_SHAPE), ICM_FEATURE_SPACE),
                    CombinedLoadHandler(StateDictLoadHandler(), InitZeroLoadHandler())).to(DEVICE)
    icm.share_memory()

    loss_func = TakeSimilarActionsLossFunction(net)

    optimizer = cman.load("opt", optim.Adam(net.parameters(), lr=LEARNING_RATE), StateDictLoadHandler())

    train_loop = cman.load("train_loop",
                           RandomReplayTrainingLoop(DISCOUNT, REPLAY_BUFFER, MIN_TRACE_TO_TRAIN, PPO_TRAIN_ROUNDS,
                                                    net.rewrite_obs, writer, DEVICE),
                           StateDictLoadHandler())

    icm_opt = ModuleCuriosityOptimizer(icm, net.rewrite_obs, action_space, CURIOSITY_LEARNING_RATE, DEVICE)

    with get_parallel_queue(num_processes=NUM_PROCESSES, episode_runner=run_mcts_episode,
                            net=net, env=env, max_length=MAX_EPISODE_LEN,
                            curiosity=icm_opt, state_observer=None, device=DEVICE,
                            c_puct=MCTS_C_PUCT, num_batches=MCTS_NUM_BATCHES,
                            batch_size=MCTS_BATCH_SIZE) as sim_round_queue:

        while True:
            train_loop(sim_round_queue, loss_func, optimizer)

            if train_loop.num_rounds % CHECKPOINT_EVERY == 0:
                cman.save({
                    "net": net,
                    "icm": icm,
                    "opt": optimizer,
                    "train_loop": train_loop
                })


if __name__ == '__main__':
    run()
