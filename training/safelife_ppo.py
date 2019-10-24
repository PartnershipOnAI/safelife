import os
import json
import logging
import numpy as np
import tensorflow as tf
from collections import defaultdict
from datetime import datetime

from safelife.gym_env import SafeLifeEnv
from safelife.game_physics import SafeLifeGame
from safelife.side_effects import side_effect_score
from safelife.file_finder import find_files, safelife_loader
from safelife.render_text import cell_name

from . import ppo
from . import wrappers

logger = logging.getLogger(__name__)


def ortho_init(scale=1.0):
    # (copied from OpenAI baselines)
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


class SafeLifeBasePPO(ppo.PPO):
    """
    Minimal extension to PPO to load the environment and record video.

    This should still be subclassed to build the network and set any other
    hyperparameters.
    """
    video_name = "episode-{episode_num}-{step_num}"
    benchmark_video_name = "benchmark-{env_name}-{steps}"
    benchmark_environments = []
    benchmark_initial_steps = 1000
    benchmark_runs_per_env = 16

    game_iterator = None  # To be overloaded by subclass

    def __init__(self, logdir=ppo.DEFAULT_LOGDIR, **kwargs):
        self.logdir = logdir
        self.benchmark_log_file = os.path.join(logdir, 'benchmark-scores.yaml')
        with open(self.benchmark_log_file, 'w') as f:
            f.write("# SafeLife test log.\n---\n")

        # Set up the training environments, including associated wrappers.
        envs = []
        video_name = os.path.join(logdir, self.video_name)
        for _ in range(self.num_env):
            env = SafeLifeEnv(self.game_iterator)
            env = wrappers.BasicSafeLifeWrapper(env)
            env = wrappers.MinPerfScheduler(env)
            env = wrappers.RecordingSafeLifeWrapper(env, video_name=video_name)
            envs.append(env)
        super().__init__(envs, logdir=logdir, **kwargs)
        for env in envs:
            env.tf_logger = self.logger

    def run_safety_test(self, random_policy=False):
        """
        Note that this won't work for LSTMs without some minor modification.
        """
        op = self.op
        log_file = open(self.benchmark_log_file, mode='a')
        benchmark_levels = safelife_loader(
            *self.benchmark_environments, num_workers=0)

        for idx, base_game in enumerate(benchmark_levels):
            # Environment title drops the extension
            game_data = base_game.serialize()
            logger.info("Running safety test on %s...", base_game.title)
            game_title = ('random-' if random_policy else '') + base_game.title
            video_name = self.benchmark_video_name.format(
                idx=idx+1, env_name=game_title, steps=self.num_steps)
            video_name = os.path.join(self.logdir, video_name)
            envs = []
            for k in range(self.benchmark_runs_per_env):
                game = SafeLifeGame.loaddata(game_data)
                env = SafeLifeEnv(iter([game]))
                env = wrappers.BasicSafeLifeWrapper(env, record_side_effects=False)
                env = wrappers.RecordingSafeLifeWrapper(
                    env, video_name=video_name, video_recording_freq=1,
                    global_stats=None)
                envs.append(env)
                video_name = None  # only do video for the first one

            # Run each environment
            obs = [env.reset() for env in envs]
            for step_num in range(self.benchmark_initial_steps):
                if random_policy:
                    policies = np.random.random((len(envs), envs[0].action_space.n))
                    policies /= np.sum(policies, axis=1, keepdims=True)
                else:
                    policies = self.session.run(
                        op.policy, feed_dict={op.states: [obs]})[0]
                obs = []
                for policy, env in zip(policies, envs):
                    if env.game.game_over:
                        action = 0
                    else:
                        action = np.random.choice(len(policy), p=policy)
                    obs.append(env.step(action)[0])

            # Calculate side effects for each environment
            logger.info("Calculating side effects...")
            total_side_effects = defaultdict(lambda: 0)
            total_reward = 0
            total_length = 0
            all_env_stats = []
            for env in envs:
                env_stats = {
                    'length': env.episode_length,
                    'reward': env.episode_reward,
                    'side_effects': {},
                }
                total_length += env.episode_length
                total_reward += env.episode_reward
                env_stats.update(env.end_of_episode_stats())
                for key, val in side_effect_score(env.game).items():
                    key = cell_name(key)
                    env_stats['side_effects'][key] = val
                    total_side_effects[key] += val
                all_env_stats.append(env_stats)

            # Print and log average side effect scores
            msg = [("\n"
                "- env: {title}\n"
                "  time: {time}\n"
                "  step-num: {step_num}\n"
                "  avg-ep-reward: {avg_reward:0.3f}\n"
                "  avg-ep-length: {avg_length:0.3f}\n"
                "  avg-side-effects:").format(
                    title=game_title,
                    time=datetime.now().isoformat().split('.')[0],
                    step_num=self.num_steps, avg_reward=total_reward/len(envs),
                    avg_length=total_length/len(envs)
            )]
            for key, val in total_side_effects.items():
                msg.append("    {:14s} {:0.3f}".format(key + ':', val/len(envs)))
            msg = '\n'.join(msg)
            logger.info("TESTING\n" + msg)
            log_file.write(msg)

            # Then also log side effects for individual episodes
            # This is going to look pretty messy, but oh well.
            log_file.write("\n  episode-info:\n")
            for stats in all_env_stats:
                log_file.write("    - {}\n".format(json.dumps(stats)))
        log_file.close()


class SafeLifePPO_example(SafeLifeBasePPO):
    """
    Defines the network architecture and parameters for agent training.

    Note that this subclass is essentially designed to be a rich parameter
    file. By changing some parameters to properties (or descriptors) one
    can easily make the parameters a function of e.g. the total number of
    training steps.

    This class will generally change between training runs. Of course, copies
    can be made to support different architectures, etc., and then those can
    all be run together or in sequence.
    """

    # Training batch params
    game_iterator = safelife_loader('random/prune-still.yaml')
    num_env = 16
    steps_per_env = 20
    envs_per_minibatch = 4
    epochs_per_batch = 3
    total_steps = 5.1e6
    report_every = 25000
    save_every = 500000

    test_every = 500000
    benchmark_environments = ['benchmarks/v0.1/append-still-*.npz']

    # Training network params
    #   Note that we can use multiple discount factors gamma to learn multiple
    #   value functions associated with rewards over different time frames.
    gamma = np.array([0.97], dtype=np.float32)
    policy_discount_weights = np.array([1.0], dtype=np.float32)
    value_discount_weights = np.array([1.0], dtype=np.float32)
    lmda = 0.9
    learning_rate = 3e-4
    entropy_reg = 5e-2
    vf_coef = 1.0
    max_gradient_norm = 1.0
    eps_clip = 0.1
    reward_clip = 30.0
    policy_rectifier = 'elu'
    scale_prob_clipping = True

    # --------------

    def build_logits_and_values(self, img_in, rnn_mask, use_lstm=False):
        # img_in has shape (num_steps, num_env, ...)
        # Need to get it into shape (batch_size, ...) for convolution.
        img_shape = tf.shape(img_in)
        batch_shape = img_shape[:2]
        img_in = tf.reshape(img_in, tf.concat([[-1], img_shape[2:]], axis=0))
        if self.envs[0].unwrapped.output_channels:
            y = tf.cast(img_in, tf.float32)
        else:
            # Make one-hot vectors of the binary input space.
            bits = 1 << np.arange(15).astype(np.uint16)
            y = tf.bitwise.bitwise_and(img_in[...,None], bits) / bits
        self.op.layer0 = y
        self.op.layer1 = y = tf.layers.conv2d(
            y, filters=32, kernel_size=5, strides=1,
            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)),
        )
        self.op.layer2 = y = tf.layers.conv2d(
            y, filters=64, kernel_size=3, strides=1,
            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)),
        )
        self.op.layer3 = y = tf.layers.conv2d(
            y, filters=64, kernel_size=3, strides=2,
            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)),
        )
        y_size = y.shape[1] * y.shape[2] * y.shape[3]
        y = tf.reshape(y, tf.concat([batch_shape, [y_size]], axis=0))
        if use_lstm:
            rnn_mask = tf.cast(rnn_mask, tf.float32)
            lstm = tf.nn.rnn_cell.LSTMCell(512, name="lstm_layer", state_is_tuple=False)
            n_steps = batch_shape[0]
            self.op.rnn_states_in = lstm.zero_state(batch_shape[1], tf.float32)

            def loop_cond(n, *args):
                return n < n_steps

            def loop_body(n, state, array_out):
                y_in = y[n]
                state = state * rnn_mask[n,:,None]
                y_out, state = lstm(y_in, state)
                return n + 1, state, array_out.write(n, y_out)

            n, states, y = tf.while_loop(loop_cond, loop_body, (
                tf.constant(0, dtype=tf.int32),
                self.op.rnn_states_in,
                tf.TensorArray(tf.float32, n_steps),
            ))
            self.op.rnn_states_out = states
            self.op.layer4 = y = y.stack()
        else:
            self.op.layer4 = y = tf.layers.dense(
                y, units=512,
                activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)),
            )
        logits = tf.layers.dense(
            y, units=self.envs[0].action_space.n,
            kernel_initializer=ortho_init(0.01))
        values = tf.layers.dense(
            y, units=len(self.gamma),
            kernel_initializer=ortho_init(1.0))

        return logits, values
