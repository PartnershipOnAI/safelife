import logging
import numpy as np
import tensorflow as tf
from scipy import interpolate

from safelife.safelife_env import SafeLifeEnv
from safelife.file_finder import SafeLifeLevelIterator

from . import ppo

logger = logging.getLogger(__name__)


def linear_schedule(t, y):
    return interpolate.UnivariateSpline(t, y, s=0, k=1, ext='const')


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


class SafeLifePPO(ppo.PPO):
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
    level_iterator = SafeLifeLevelIterator('random/prune-still-easy.yaml')
    video_name = "episode-{episode_num}-{step_num}"
    num_env = 16
    steps_per_env = 20
    envs_per_minibatch = 4
    epochs_per_batch = 3
    total_steps = 5.1e6
    report_every = 25000
    save_every = 500000

    # Training environment params
    impact_penalty = 0.0
    min_performance = linear_schedule([0.5e6, 1.5e6], [0.01, 0.3])

    # Training network params
    #   Note that we can use multiple discount factors gamma to learn multiple
    #   value functions associated with rewards over different time frames.
    gamma = 0.97
    lmda = 0.9
    learning_rate = 3e-4
    entropy_reg = 5e-2
    vf_coef = 1.0
    max_gradient_norm = 1.0
    eps_clip = 0.1
    reward_clip = 30.0
    policy_rectifier = 'relu'
    scale_prob_clipping = True

    # --------
    # A few functions to keep episode and step counters synced:

    def restore_checkpoint(self, logdir, raise_on_error=False):
        success = super().restore_checkpoint(logdir, raise_on_error)
        num_steps, num_episodes = self.session.run(
            [self.op.num_steps, self.op.num_episodes])
        SafeLifeEnv.global_counter.episodes_started = num_episodes
        SafeLifeEnv.global_counter.episodes_completed = num_episodes
        SafeLifeEnv.global_counter.num_steps = num_steps
        return success

    @property
    def num_episodes(self):
        # Override the num_episodes attribute to always point to
        # the global counter on SafeLifeEnv. This ensures that it
        # increases even when using the ContinuingEnv wrapper.
        return SafeLifeEnv.global_counter.episodes_completed

    @num_episodes.setter
    def num_episodes(self, val):
        pass  # don't allow setting directly, but don't throw an error

    # --------
    # Building network architecture:

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
            y, filters=32, kernel_size=5, strides=2,
            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)),
        )
        self.op.layer2 = y = tf.layers.conv2d(
            y, filters=64, kernel_size=3, strides=2,
            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)),
        )
        self.op.layer3 = y = tf.layers.conv2d(
            y, filters=64, kernel_size=3, strides=1,
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
            y, units=1,
            kernel_initializer=ortho_init(1.0))

        return logits, values[...,0]
