import numpy as np
import tensorflow as tf

from safelife.safety_gym import GameOfLifeEnv
from . import ppo


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


class GameOfLifePPO(ppo.PPO):
    video_freq = 20
    num_env = 16
    gamma = np.array([0.9, 0.99], dtype=np.float32)
    policy_discount_weights = np.array([0.5, 0.5], dtype=np.float32)
    value_discount_weights = np.array([0.5, 0.5], dtype=np.float32)
    lmda = 0.9
    learning_rate = 3e-4
    entropy_reg = 1e-2
    vf_coef = 1.0
    max_gradient_norm = 1.0
    eps_clip = 0.1
    reward_clip = 10.0
    policy_rectifier = 'elu'
    scale_prob_clipping = True

    def __init__(self, **kwargs):
        super().__init__(GameOfLifeEnv, **kwargs)

    def build_logits_and_values(self, img_in, cell_mask, use_lstm=False):
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
            cell_mask = tf.cast(cell_mask, tf.float32)
            lstm = tf.nn.rnn_cell.LSTMCell(512, name="lstm_layer", state_is_tuple=False)
            n_steps = batch_shape[0]
            self.op.cell_states_in = lstm.zero_state(batch_shape[1], tf.float32)

            def loop_cond(n, *args):
                return n < n_steps

            def loop_body(n, state, array_out):
                y_in = y[n]
                state = state * cell_mask[n,:,None]
                y_out, state = lstm(y_in, state)
                return n + 1, state, array_out.write(n, y_out)

            n, states, y = tf.while_loop(loop_cond, loop_body, (
                tf.constant(0, dtype=tf.int32),
                self.op.cell_states_in,
                tf.TensorArray(tf.float32, n_steps),
            ))
            self.op.cell_states_out = states
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

        def dead_fraction(x):
            x = tf.equal(x, 0.0)
            x = tf.cast(x, tf.float32)
            return tf.reduce_mean(x)

        with tf.name_scope('is_dead'):
            tf.summary.scalar('layer1', dead_fraction(self.op.layer1))
            tf.summary.scalar('layer2', dead_fraction(self.op.layer2))
            tf.summary.scalar('layer3', dead_fraction(self.op.layer3))
            tf.summary.scalar('layer4', dead_fraction(self.op.layer4))

        return logits, values


if __name__ == '__main__':
    model = GameOfLifePPO()
    model.train(5e7)
