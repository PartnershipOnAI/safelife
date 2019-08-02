import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d

from safelife.game_physics import CellTypes
from .safelife_ppo import SafeLifePPO, ortho_init


class SafeLifeAUP(SafeLifePPO):
    aup_num_rewards = 100
    aup_filter_size = 5
    aup_discount = 0.9
    aup_penalty_schedule = interp1d([0, 2e5, 2e6, 1e20], [0, 0, 0.05, 0.05])
    aup_channels = [CellTypes.alive_bit, CellTypes.frozen_bit]
    aup_learning_rate = 1e-3
    aup_clip1 = 5.0  # clipping of qfunc training inputs
    aup_clip2 = 1.0  # clipping of penalty from qfunc
    # test_every = 0

    @property
    def aup_penalty_coef(self):
        return float(self.aup_penalty_schedule(self.num_steps))

    def build_graph(self):
        super().build_graph()
        op = self.op

        self.aup_filters = np.random.randn(
            self.aup_filter_size *
            self.aup_filter_size *
            len(self.aup_channels),
            self.aup_num_rewards,
        ).astype(np.float32)
        # Normalize the filters such that the sum of all weights in a single
        # filter has zero mean and unit deviation.
        self.aup_filters -= np.average(self.aup_filters, axis=0)
        self.aup_filters /= np.sqrt(np.sum(self.aup_filters**2, axis=0))
        self.aup_filters = self.aup_filters.reshape(
            self.aup_filter_size,
            self.aup_filter_size,
            len(self.aup_channels),
            self.aup_num_rewards,
        )

        with tf.name_scope("aup_utility"):
            n = self.aup_filter_size
            op.full_boards = tf.placeholder(tf.int16, [None, None, None], name="board")
            op.aup_filter = tf.constant(self.aup_filters, name="filters")
            # Add padding such that the boards wrap
            boards = op.full_boards
            boards = tf.concat([boards, boards[:, :n-1]], axis=1)
            boards = tf.concat([boards, boards[:, :, :n-1]], axis=2)
            # Get rid of the agent
            agent_mask = tf.bitwise.bitwise_and(boards, CellTypes.agent) // CellTypes.agent
            boards *= 1 - agent_mask
            op.aup_board = boards  # TEMPORARY
            # Extract the bits we care about and cast to float
            bits = 1 << np.array(self.aup_channels).astype(np.uint16)
            boards = tf.bitwise.bitwise_and(boards[...,None], bits) / bits
            # Each cell of the board should then be in the set {-1, 1} such that
            # the convolution output is normally distributed.
            boards = 2 * boards - 1
            # Convolve the boards with our random filters.
            conv = tf.nn.conv2d(boards, op.aup_filter, strides=[1]*4, padding="VALID")
            # Square the result (to apply a nonlinearity) and take the sum
            # to get the total utility for each reward channel.
            # Note this should have an approximately χ^2 distribution with
            # degrees of freedom (and therefore mean) equal to total board size.
            # However, the *difference* between the utility for two different
            # actions should be of order the size of the area affected by that
            # action, which is to say 1.
            op.aup_utility = tf.reduce_sum(tf.square(conv), axis=[1,2])

        with tf.name_scope("aup_q_funcs"):
            op.aup_q_funcs = self.build_aup_q_funcs(op.states)  # shape [None, None, num_rewards, num_actions]
            q_func1 = tf.reduce_sum(op.hot_actions[..., None, :] * op.aup_q_funcs, axis=-1)
            q_func0 = op.aup_q_funcs[..., 0]
            q_diff = q_func1 - q_func0
            op.aup_penalty = tf.reduce_mean(tf.abs(q_diff), axis=-1)  # shape (None, None)

            tf.summary.scalar('q_avg', tf.reduce_mean(op.aup_q_funcs))
            tf.summary.histogram('q_func', op.aup_q_funcs)
            tf.summary.histogram('q_diff', q_diff)
            tf.summary.scalar("avg_penalty", tf.reduce_mean(op.aup_penalty))
            tf.summary.histogram("penalty", op.aup_penalty)
        op.summary = tf.summary.merge_all()

        with tf.name_scope("aup_training"):
            op.aup_mask = tf.placeholder(tf.float32, [None, None], name="aup_mask")
            op.aup_reward = tf.placeholder(tf.float32, [None, None, self.aup_num_rewards], name="aup_reward")
            op.aup_discount = tf.constant(self.aup_discount, name="aup_discount")
            op.aup_learning_rate = tf.constant(self.aup_learning_rate, name="aup_learning_rate")
            q_func = tf.reduce_sum(op.hot_actions[..., None, :] * op.aup_q_funcs[:-1], axis=-1)
            v_func = tf.reduce_max(op.aup_q_funcs[1:], axis=-1)
            next_q = tf.stop_gradient(op.aup_reward + op.aup_discount * v_func)
            loss = tf.square(q_func - next_q) * op.aup_mask[..., None]
            loss = tf.reduce_mean(loss)
            optimizer = self.build_optimizer(op.aup_learning_rate)
            op.aup_train = optimizer.minimize(loss, name="aup_train"),
            op.aup_summary = tf.summary.merge([
                tf.summary.scalar("aux_reward", tf.sqrt(tf.reduce_mean(tf.square(op.aup_reward)))),
                tf.summary.histogram("aux_reward", op.aup_reward),
            ])


    def build_aup_q_funcs(self, img_in):
        # Make this basically identical to the network for the normal value
        # functions.
        img_shape = tf.shape(img_in)
        batch_shape = img_shape[:2]
        img_in = tf.reshape(img_in, tf.concat([[-1], img_shape[2:]], axis=0))
        if self.envs[0].unwrapped.output_channels:
            y = tf.cast(img_in, tf.float32)
        else:
            # Make one-hot vectors of the binary input space.
            bits = 1 << np.arange(15).astype(np.uint16)
            y = tf.bitwise.bitwise_and(img_in[...,None], bits) / bits
        y = tf.layers.conv2d(
            y, filters=32, kernel_size=5, strides=1,
            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)),
        )
        y = tf.layers.conv2d(
            y, filters=64, kernel_size=3, strides=1,
            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)),
        )
        y = tf.layers.conv2d(
            y, filters=64, kernel_size=3, strides=2,
            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)),
        )
        y_size = y.shape[1] * y.shape[2] * y.shape[3]
        y = tf.reshape(y, tf.concat([batch_shape, [y_size]], axis=0))
        y = tf.layers.dense(
            y, units=self.aup_num_rewards*9,
            kernel_initializer=ortho_init(0.01))
        y = tf.reshape(y, tf.concat([batch_shape, [self.aup_num_rewards, 9]], axis=0))
        return y

    def run_agents(self, num_steps):
        op = self.op
        session = self.session

        batch = super().run_agents(num_steps)

        # First, calculate the utility for each board to update the q-functions
        # Note that we're dropping the utility for the first step in each batch
        # and the first and last step of each episode where the observations
        # states and full board states don't line up.
        boards = [x['board'] for x in batch.info.ravel()]
        aup_utility = session.run(op.aup_utility, feed_dict={op.full_boards:boards})
        aup_utility = aup_utility.reshape(batch.info.shape + aup_utility.shape[1:])
        delta_utility = aup_utility[1:] - aup_utility[:-1]  # shape (num_steps-1, num_env, num_aux)
        delta_utility = np.clip(delta_utility, -self.aup_clip1, self.aup_clip1)
        mask = ~(batch.end_episode[1:] | batch.end_episode[:-1])    # shape (num_steps-1, num_env)
        session.run(op.aup_train, feed_dict={
            op.states: batch.states[1:],  # shape (num_steps, num_env, ...)
            op.actions: batch.actions[1:],  # shape (num_steps-1, num_env)
            op.aup_mask: mask,
            op.aup_reward: delta_utility
        })

        self.logger.add_summary(
            session.run(op.aup_summary, feed_dict={op.aup_reward:delta_utility}),
            self.num_steps)

        # Then modify the rewards to take into account AUP
        aup_penalty = session.run(op.aup_penalty, feed_dict={
            op.states: batch.states[:-1],
            op.actions: batch.actions
        })
        aup_penalty = np.clip(aup_penalty, -self.aup_clip2, self.aup_clip2)
        return batch._replace(rewards=batch.rewards - self.aup_penalty_coef * aup_penalty)
