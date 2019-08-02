import os
import numpy as np
import tensorflow as tf

from safelife.gym_env import SafeLifeEnv
from safelife.side_effects import policy_side_effect_score
from safelife.gen_board import region_population_params
from . import ppo
from .wrappers import SafeLifeWrapper


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
    video_freq = 100
    video_counter = None      # counter for filenames to save videos in
    video_name = "episode-{episode}-{steps}"
    test_video_name = "test-{env_name}-{steps}"
    test_environments = []

    environment_params = {}
    board_gen_params = {}
    side_effect_args = {}

    def __init__(self, logdir=ppo.DEFAULT_LOGDIR, **kwargs):
        self.logdir = logdir
        envs = [
            SafeLifeWrapper(
                SafeLifeEnv(**self.environment_params),
                reset_callback=self.update_environment,
            ) for _ in range(self.num_env)
        ]
        super().__init__(envs, logdir=logdir, **kwargs)

    def update_environment(self, env_wrapper):
        # Called just before an environment resets
        if self.video_counter is None:
            self.video_counter = self.num_episodes
        if self.video_freq > 0 and self.video_counter % self.video_freq == 0:
            base_name = self.video_name.format(
                episode=self.video_counter, steps=self.num_steps)
            env_wrapper.video_name = os.path.join(self.logdir, base_name)
        else:
            env_wrapper.video_name = None
        self.video_counter += 1
        # If the board_gen_params are implemented as a property, then they
        # could easily be changed with every update to do some sort of
        # curriculum learning.
        env_wrapper.unwrapped.board_gen_params = self.board_gen_params

    def run_safety_test(self):
        op = self.op

        def policy(obs, memory):
            fd = {op.states: [[obs]]}
            if memory is not None:
                fd[op.rnn_states_in] = memory
            if op.rnn_states_out is not None:
                policy, memory = self.session.run(
                    [op.policy, op.rnn_states_out], feed_dict=fd)
            else:
                policy = self.session.run(op.policy, feed_dict=fd)
            policy = policy[0, 0]
            return np.random.choice(len(policy), p=policy), memory

        for idx, env_name in enumerate(self.test_environments):
            env = SafeLifeEnv(
                fixed_levels=[env_name], **self.environment_params)
            env.reset()
            video_name = os.path.join(self.logdir, self.test_video_name.format(
                idx=idx+1, env_name=env.unwrapped.state.title,
                steps=self.num_steps))
            env = SafeLifeWrapper(
                env, video_name=video_name, on_name_conflict="abort")
            safety_scores, avg_reward, avg_length = policy_side_effect_score(
                policy, env, named_keys=True, **self.side_effect_args)

            # ...somehow record this. For now just print.
            # Might want to just output to a dedicated log file.
            print("\nTesting", env.unwrapped.state.title, self.num_steps)
            print("    Episode reward", avg_reward)
            print("    Episode length", avg_length)
            print("Side effects:")
            for key, val in safety_scores.items():
                print("    {:14s} {:0.3f}".format(key, val))
            print("")


def make_curriculum():

    levels = []
    taught_unmaking = False
    for x in np.linspace(1, 7, 13):
        # before we mix in destroy tasks, train on some tasks 
        # that are just destroying things
        if region_population_params(x)["region_types"]["destroy"] > 0:
            if not taught_unmaking:
                taught_unmaking = True
                levels.append({
                    'board_shape': (25, 25),
                    'difficulty': 1,
                    'max_regions': 4,
                    'region_types': {
                        'build': 0,
                        'destroy': 1
                    }
                    })

        levels.append({
            'board_shape': (25, 25),
            'difficulty': x,
            'max_regions': 4,
            'start_region': None
            })
    return levels


class SafeLifePPO(SafeLifeBasePPO):
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
    num_env = 16
    steps_per_env = 20
    envs_per_minibatch = 4
    epochs_per_batch = 3
    total_steps = 5e6
    report_every = 5000
    save_every = 50000

    test_every = 500000
    test_environments = ['benchmarks/test-append.npz']

    # Training network params
    gamma = np.array([0.9, 0.99], dtype=np.float32)
    policy_discount_weights = np.array([0.5, 0.5], dtype=np.float32)
    value_discount_weights = np.array([0.5, 0.5], dtype=np.float32)
    lmda = 0.9
    learning_rate = 3e-4
    entropy_reg = 5e-2
    vf_coef = 1.0
    max_gradient_norm = 1.0
    eps_clip = 0.1
    reward_clip = 10.0
    policy_rectifier = 'elu'
    scale_prob_clipping = True

    # Environment params
    environment_params = {
        'max_steps': 1200,
        'movement_bonus': 0.04,
        'movement_bonus_power': 0.01,
        'remove_white_goals': True,
        'view_shape': (15, 15),
        'output_channels': tuple(range(15)),
    }
    board_gen_params = {
        'board_shape': (25, 25),
        'difficulty': 3.9,
        'max_regions': 4,
        'region_types': {
            # 'destroy': 1,
            # 'prune': 2,
            'build': 1,
            'append': 2,
        },
        'start_region': None,
    }

    curriculum_params = make_curriculum()
    # --------------

    def update_environment(self, env_wrapper):
        "Customized variant of update_enviroment to implement curricular learning."
        super().update_enviroment(env_wrapper)
        env = env_wrapper.unwrapped
        self.adjust_to_current_curriculum(env)

    def adjust_to_current_curriculum(self, env):
        "Modify an environment to fit with the current curriculum stage."


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
