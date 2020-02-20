import os
import gym
import numpy as np
# import tensorflow as tf
import torch

from safelife.safelife_env import SafeLifeEnv
from safelife.safelife_game import CellTypes
from safelife.file_finder import SafeLifeLevelIterator
from safelife import env_wrappers

t = torch.nn
s = t.Sequential

c = 'circular'  # https://github.com/pytorch/pytorch/pull/17240
                # note that padding must be kernel size - 1

class EmbeddingNetwork(t.Module):

    def __init__(self, conf):
        super().__init__()
        ksize = conf.conv1kernel
        self.embedding = s(
            t.Conv2d(10, 32, ksize, stride=1, padding=ksize-1, padding_mode=c),
            t.ReLU(),
            t.Conv2d(32, conf.embedding_depth, ksize, stride=1, padding=ksize-1, padding_mode=c),
            t.ReLU())

    def forward(self, x):
        x = x.transpose(-1, -3) # convert from SafeLife observation to torch
        return self.embedding(x)


class PolicyNetwork(t.Module):

    def __init__(self, conf):
        # todo: figure out how much layer reuse we really want here
        ksize = conf.conv2kernel
        chans = conf.embedding_depth
        self.policy_value = [
             t.Conv2d(chans, chans, ksize, padding=ksize-1, stride=1, padding_mode=c),
             t.ReLU(),
             t.Linear(conf.hidden_size),
             t.ReLU(),
             t.Linear(conf.hidden_size, 1)]

        self.policy = self.policy_value[0:4] + [
            # t.Linear(conf.hidden_size), t.ReLU(),
            t.Softmax(len(conf.action_space))
        ]

        self.policy_value = s(self.policy_value)


class DynamicsNetwork(t.Module):
    "This is hardcoded due to artistic disagreements with this codebase's layout :)"
    def __init__(self, conf):
        super().__init__()
        self.linear_inp = t.Linear(np.product(conf.embedding_shape) + len(conf.action_space),
                                   conf.global_dense_embedding_size)

        # channels is the embedding size plus the amount of global state (like
        # "move left") we are feeding to the conv layers

        ksize = conf.conv2kernel
        chans = conf.embedding_depth + conf.global_dense_embedding_size
        self.conv1 =s(
              t.Conv2d(chans, chans, ksize, stride=1, padding=ksize-1, padding_mode=c),
              t.ReLU())
        self.conv2 = s(
              t.Conv2d(chans, conf.embedding_depth, ksize, stride=1, padding=ksize-1, padding_mode=c),
              t.ReLU())
        conv_shape = conf.embedding_shape[:-1] + (chans,)

        self.reward = s(
            t.Linear(np.product(conv_shape),128),
            t.ReLU(),
            t.Linear(128, 1))

    def forward(self, x, action):
        # TODO ensure that action is 1-hot
        global_inp = torch.cat((x.flatten(start_dim=1), action), dim=1)
        global_view = self.linear_inp(global_inp)
        global_view = global_view[..., np.newaxis, np.newaxis]
        global_view = global_view.expand(x.shape[0], global_view.shape[1], x.shape[2], x.shape[3])
        amended_state = torch.cat((x, global_view), dim=1)

        x = self.conv1(amended_state)
        reward = self.reward(x.flatten(start_dim=1))
        x = self.conv2(x)

        return x, reward


class MuZeroConfig:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game

        ### Game
        self.observation_shape = 10  # Dimensions of the game observation
        self.action_space = SafeLifeEnv.action_names  # Fixed list of all possible actions
        self.view_shape = (26, 26)


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
        self.hidden_size = 512
        self.global_dense_embedding_size = 16
        self.conv1kernel = 5
        self.conv2kernel = 3

        # hidden representations
        self.embedding_depth = 64
        self.embedding_shape = self.view_shape + (self.embedding_depth,) # for SafeLife we're helping the agent by giving it an internal representation that matches the game's state space


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


def Game(conf, seed=None, logdir="./safelife-logs"):
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
        view_shape=conf.view_shape,
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
    # env = env_wrappers.RecordingSafeLifeWrapper(
    #     env, video_name=video_name, tf_logger=tf_logger,
    #     log_file=episode_log)
    env = env_wrappers.ExtraExitBonus(env)
    return env
