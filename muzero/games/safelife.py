import os
import gym
import numpy
# import tensorflow as tf
import torch

from safelife.safelife_env import SafeLifeEnv
from safelife.safelife_game import CellTypes
from safelife.file_finder import SafeLifeLevelIterator
from safelife import env_wrappers


class SafelifeConvNetork(torch.nn.Module):
    "This is hardcoded due to artistic disagreements with this codebase's layout :)"
    def __init__(self):
        embedding_layer1 = torch.nn.Conv2d(10, 32, 5, stride=2)
        embedding_layer2 = torch.nn.Conv2d(32, 64, 3, stride=2)

        dynamics_layer1 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        dynamics_mid = FullyConnectedNetwork(...)
        dynamics_layer2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)

        prediction_layer1 = torch.nn.Conv2d(64, 64, 3, stride=1)
        prediction_layer2 = FullyConnectedNetwork(512)
        prediction_layer3 = SoftMaxNetwork(9)

        reward_layer1 = torch.nn.Conv2d(64, 64, 3, stride=1)
        reward_layer2 = FullyConnectedNetwork(512)
        reward_layer3 = FullyConnectedNetwork(1)

        value_layer1 = torch.nn.Conv2d(64, 64, 3, stride=1)
        value_layer2 = FullyConnectedNetwork(512)
        value_layer3 = FullyConnectedNetwork(1)

        # share some of these layers across functions

    def forward(self, x):
        ...


class MuZeroConfig:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game

        ### Game
        self.observation_shape = 10  # Dimensions of the game observation
        self.action_space = SafeLifeEnv.action_names  # Fixed list of all possible actions


        ### Self-Play
        self.num_actors = 1  # Number of simultaneous threads self-playing to feed the replay buffer
        self.max_moves = 500  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of futur moves self-simulated
        self.discount = 0.97  # Chronological discount of the reward
        self.self_play_delay = None # Number of seconds to wait after each played game to adjust the self play / training ratio to avoid overfitting (Recommended is 13:1 see https://arxiv.org/abs/1902.04522 Appendix A)

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.encoding_size = 64
        self.hidden_size = 32

        # Training
        self.results_path = "./pretrained"  # Path to store the model weights
        self.training_steps = 2000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.checkpoint_interval = 10  # Number of training steps before using the model for sef-playing
        self.window_size = 1000  # Number of self-play games to keep in the replay buffer
        self.td_steps = 10  # Number of steps in the futur to take into account for calculating the target value
        self.training_delay = 1 # Number of seconds to wait after each training to adjust the self play / training ratio to avoid overfitting (Recommended is 13:1 see https://arxiv.org/abs/1902.04522 Appendix A)

        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9

        # Test
        self.test_episodes = 2  # Number of game played to evaluate the network

        # Exponential learning rate schedule
        self.lr_init = 0.0005  # Initial learning rate
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 3500

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.25 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


def Game(seed=None, logdir="./safelife-logs"):
    """
    if logdir:
        video_name = os.path.join(logdir, "episode-{episode_num}-{step_num}")
    else:
        video_name = None

    if logdir:
        fname = os.path.join(logdir, "training.yaml")
        if os.path.exists(fname):
            episode_log = open(fname, 'a')
        else:
            episode_log = open(fname, 'w')
            episode_log.write("# Training episodes\n---\n")
    else:
        episode_log = None

    tf_logger = tf.summary.FileWriter(logdir)
    """

    levelgen = SafeLifeLevelIterator('random/append-still-easy.yaml')
    env = SafeLifeEnv(
        levelgen,
        view_shape=(25,25),
        output_channels=(
            CellTypes.alive_bit,
            CellTypes.agent_bit,
            CellTypes.pushable_bit,
            CellTypes.destructible_bit,
            CellTypes.frozen_bit,
            CellTypes.spawning_bit,
            CellTypes.exit_bit,
            CellTypes.color_bit + 0,  # red
            CellTypes.color_bit + 1,  # green
            CellTypes.color_bit + 5,  # blue goal
        ))
    env.seed(seed)
    env = env_wrappers.MovementBonusWrapper(env, as_penalty=True)
    env = env_wrappers.MinPerformanceScheduler(env, min_performance=0.1)
    #env = env_wrappers.RecordingSafeLifeWrapper(
    #    env, video_name=video_name, tf_logger=tf_logger,
    #    log_file=episode_log)
    env = env_wrappers.ExtraExitBonus(env)
    return env
