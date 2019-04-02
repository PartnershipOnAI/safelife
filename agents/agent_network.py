import numpy as np
import tensorflow as tf

from safety_net.safety_gym import GameOfLifeEnv
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
    num_env = 4
    gamma = 0.99
    lmda = 0.95
    learning_rate = 3e-4
    entropy_reg = 0.0
    vf_coef = 1.0
    max_gradient_norm = 1.0
    eps_clip = 0.1
    reward_clip = 10.0

    def __init__(self, **kwargs):
        super().__init__(GameOfLifeEnv, **kwargs)

    def build_logits_and_values(self, img_in):
        y = tf.cast(img_in, tf.float32)
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
        self.op.layer4 = y = tf.layers.dense(
            tf.layers.flatten(y), units=512,
            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)),
        )
        logits = tf.layers.dense(
            y, units=self.envs[0].action_space.n,
            kernel_initializer=ortho_init(0.01))
        values = tf.layers.dense(
            y, units=1,
            kernel_initializer=ortho_init(1.0))[:,0]

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
