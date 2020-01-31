"""
Algorithm for Proximal Policy Optimization.
"""

import os
import logging
import inspect
from types import SimpleNamespace
from collections import namedtuple
from functools import wraps

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def named_output(*names):
    """
    A simple decorator to transform a function's output to a named tuple.
    """
    def decorator(func):
        rtype = namedtuple(func.__name__ + '_rval', names)

        @wraps(func)
        def wrapped(*args, **kwargs):
            rval = func(*args, **kwargs)
            if isinstance(rval, tuple):
                rval = rtype(*rval)
            return rval
        return wrapped

    return decorator


def shuffle_arrays_in_place(*data):
    """
    This runs np.random.shuffle on multiple inputs, shuffling each in place
    in the same order (assuming they're the same length).
    """
    rng_state = np.random.get_state()
    for x in data:
        np.random.set_state(rng_state)
        np.random.shuffle(x)


def shuffle_arrays(*data):
    # Despite the nested for loops, this is actually a little bit faster
    # than the above because it doesn't involve any copying of array elements.
    # When the array elements are large (like environment states),
    # that overhead can be large.
    idx = np.random.permutation(len(data[0]))
    return [[x[i] for i in idx] for x in data]


def eps_relu(x, eps):
    return tf.maximum(x, -eps)


def eps_elu(x, eps):
    return eps * tf.nn.elu(x / eps)


class SummaryWriter(tf.summary.FileWriter):
    # Temporary shim to make it look like a tensorboardX.SummaryWriter

    def add_scalar(self, tag, val, steps):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=val)
        self.add_summary(summary, steps)


class PPO(object):
    """
    Proximal policy optimization.

    Note that essentially all of these attributes can get overridden by
    subclasses, so the defaults set here are basically just for example.

    Note also that nothing in this class requires or assumes anything about
    SafeLife or the rest of the code base. It should be pretty generic.

    Attributes
    ----------
    gamma : float
        Discount factor for rewards.
    lmda : float
        Discount factor for generalized advantage estimator.
    vf_coef : float
        Overall coefficient of the value loss in the total loss function.
    learning_rate : float
    entropy_reg : float
    entropy_clip : float
        Used in entropy regularization. The regularization effectively doesn't
        turn on until the entropy drops below this level.
    max_gradient_norm : float
    eps_clip : float
        The PPO clipping for both policy and value losses. Note that this
        implies that the value function has been scaled to roughly unit value.
    rescale_policy_eps : bool
        If true, the policy clipping is scaled by ε → (1-π)ε
    min_eps_rescale : float
        Sets a lower bound on how much `eps_clip` can be scaled by.
        Only relevant if `rescale_policy_eps` is true.
    reward_clip : float
        Clip absolute rewards to be no larger than this.
        If zero, no clipping occurs.
    policy_rectifier : str
        One of ['relu', 'elu'].
    """
    gamma = 0.99
    lmda = 0.95  # generalized advantage estimation parameter

    learning_rate = 1e-4
    entropy_reg = 0.01
    entropy_clip = 1.0  # don't start regularization until it drops below this
    vf_coef = 0.5
    max_gradient_norm = 5.0
    eps_clip = 0.2  # PPO clipping for both value and policy losses
    rescale_policy_eps = False
    min_eps_rescale = 1e-3  # Only relevant if rescale_policy_eps = True
    reward_clip = 0.0
    policy_rectifier = 'relu'  # or 'elu' or ...more to come

    num_env = 16
    steps_per_env = 20
    envs_per_minibatch = 4
    epochs_per_batch = 3
    total_steps = 5e6
    report_every = 5000
    save_every = 10000
    record_histograms = False  # histograms take a lot of disk space; disable by default

    logdir = None
    summary_writer = None

    def __init__(self, saver_args={}, **kwargs):
        for key, val in kwargs.items():
            if (not key.startswith('_') and hasattr(self, key) and
                    not inspect.ismethod(getattr(self, key))):
                setattr(self, key, val)
            else:
                raise ValueError("Unrecognized parameter: '%s'" % (key,))

        self.op = SimpleNamespace()
        self.num_steps = 0
        self.num_episodes = 0
        self.session = tf.Session()
        if self.logdir:
            self.summary_writer = SummaryWriter(self.logdir)
        self.envs = [self.environment_factory() for _ in range(self.num_env)]
        self.build_graph()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(**saver_args)
        if self.logdir:
            self.summary_writer.add_graph(self.session.graph)
            self.restore_checkpoint(self.logdir)

    def environment_factory(self):
        """
        Factory for building a OpenAI gym-like environment.

        To be implemented by subclasses.
        """
        raise NotImplementedError

    def save_checkpoint(self):
        if self.logdir:
            logger.info("Saving new checkpoint. %i episodes, %i steps.",
                        self.num_episodes, self.num_steps)
            self.op.num_steps.load(self.num_steps, self.session)
            self.op.num_episodes.load(self.num_episodes, self.session)
            save_path = os.path.join(self.logdir, 'model')
            self.saver.save(self.session, save_path, self.num_steps)

    def restore_checkpoint(self, logdir, raise_on_error=False):
        """
        Resume training from the specified directory.

        If the directory is empty, don't load.
        """
        # Annoyingly, tf.train.latest_checkpoint fails if the directory
        # has changed. Instead, load up from the current directory so that
        # we're able to rerun training locally that was started remotely.
        import re
        checkpoint_path = os.path.join(logdir, 'checkpoint')
        if not os.path.exists(checkpoint_path):
            return False
        with open(checkpoint_path) as checkpoint_file:
            line = checkpoint_file.readline()
        match = re.match(r'.*"(.+)"', line)
        if not match:
            return False
        last_checkpoint = os.path.split(match.group(1))[1]
        last_checkpoint = os.path.join(logdir, last_checkpoint)
        try:
            self.saver.restore(self.session, last_checkpoint)
        except ValueError:
            if raise_on_error:
                raise
            else:
                return False
        self.num_steps, self.num_episodes = self.session.run(
            [self.op.num_steps, self.op.num_episodes])
        logger.info("Restoring old checkpoint. %i episodes, %i steps.",
                    self.num_episodes, self.num_steps)
        return True

    def build_graph(self):
        op = self.op
        input_space = self.envs[0].observation_space
        op.states = tf.placeholder(input_space.dtype, [None, None] + list(input_space.shape), name="state")
        op.actions = tf.placeholder(tf.int32, [None, None], name="actions")
        op.old_policy = tf.placeholder(tf.float32, [None, None], name="old_policy")
        op.returns = tf.placeholder(tf.float32, [None, None], name="returns")
        op.advantages = tf.placeholder(tf.float32, [None, None], name="advantages")
        op.old_value = tf.placeholder(tf.float32, [None, None], name="old_value")
        op.learning_rate = tf.constant(self.learning_rate, name="learning_rate")
        op.eps_clip = tf.constant(self.eps_clip, name="eps_clip")
        op.rnn_mask = tf.fill(tf.shape(op.states)[:2], True, name="rnn_mask")
        op.num_steps = tf.get_variable('num_steps', initializer=tf.constant(0))
        op.num_episodes = tf.get_variable('num_episodes', initializer=tf.constant(0))

        with tf.name_scope("policy"):
            op.rnn_states_in = None
            op.rnn_states_out = None
            op.logits, op.v = self.build_logits_and_values(op.states, op.rnn_mask)
            op.policy = tf.nn.softmax(op.logits)
            num_actions = op.policy.shape[-1].value
        op.hot_actions = tf.one_hot(op.actions, num_actions, dtype=tf.float32)
        with tf.name_scope("policy_loss"):
            a_policy = tf.reduce_sum(op.policy * op.hot_actions, axis=-1)
            prob_diff = tf.sign(op.advantages) * (1 - a_policy / op.old_policy)
            if self.rescale_policy_eps:
                # Scaling the clipping by 1 - old_policy ensures that
                # the clipping is active even when the new policy is 1.
                # This is non-standard.
                eps = op.eps_clip * (1 + self.min_eps_rescale - op.old_policy)
            else:
                eps = op.eps_clip
            rectifier = {
                'relu': eps_relu,
                'elu': eps_elu,
            }[self.policy_rectifier]
            policy_loss = tf.abs(op.advantages) * rectifier(prob_diff, eps)
            policy_loss = tf.reduce_mean(policy_loss)
        with tf.name_scope("entropy"):
            op.entropy = tf.reduce_sum(
                -op.policy * tf.log(op.policy + 1e-12), axis=-1)
            mean_entropy = tf.reduce_mean(op.entropy)
            # The entropy loss encourages higher entropy in the policy, which
            # in turn encourages more exploration.
            entropy_loss = tf.minimum(mean_entropy, self.entropy_clip)
            entropy_loss *= -self.entropy_reg
        with tf.name_scope("value_loss"):
            v_clip = op.old_value + tf.clip_by_value(
                op.v - op.old_value, -op.eps_clip, op.eps_clip)
            value_loss = tf.maximum(
                tf.square(op.v - op.returns), tf.square(v_clip - op.returns))
            value_loss = 0.5 * tf.reduce_mean(value_loss)

        with tf.name_scope("trainer"):
            total_loss = policy_loss + value_loss * self.vf_coef + entropy_loss
            optimizer = self.build_optimizer(op.learning_rate)
            grads, variables = zip(*optimizer.compute_gradients(total_loss))
            op.grads = grads
            if self.max_gradient_norm > 0:
                grads2, _ = tf.clip_by_global_norm(grads, self.max_gradient_norm)
            op.train = optimizer.apply_gradients(zip(grads2, variables))

        with tf.name_scope("rollouts"):
            tf.summary.scalar("returns", tf.reduce_mean(op.returns))
            tf.summary.scalar("advantages", tf.reduce_mean(op.advantages))
            tf.summary.scalar("values", tf.reduce_mean(op.v))
            if self.record_histograms:
                tf.summary.histogram("returns", op.returns)
                tf.summary.histogram("advantages", op.advantages)
                tf.summary.histogram("values", op.v)
        tf.summary.scalar("entropy", mean_entropy)
        if self.record_histograms:
            tf.summary.histogram("entropy", op.entropy)
            with tf.name_scope("losses"):
                tf.summary.histogram("gradients", tf.global_norm(grads))
                tf.summary.histogram("policy_loss", policy_loss)
                tf.summary.histogram("value_loss", value_loss)
        op.summary = tf.summary.merge_all()

    def build_optimizer(self, learning_rate):
        return tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            epsilon=1e-6,
        )

    def build_logits_and_values(self, states):
        """
        Operations for creating policy logits and value functions.

        There should be an equal number of logits and possible actions.

        If the policy function uses an RNN, it should store the input and
        output cell states in self.op.rnn_states_in and rnn_states_out.

        To be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    def rnn_zero_state(self):
        if self.op.rnn_states_in is not None:
            return np.zeros(
                self.op.rnn_states_in.shape[1:].as_list(),
                dtype=self.op.rnn_states_in.dtype.as_numpy_dtype)
        else:
            return None

    def policy(self, obs, rnn_state=None):
        """
        Probability of taking each action given an observation.
        """
        op = self.op
        session = self.session

        # Note that the op.states operator expects two dimensions for the
        # observations: one for the number of environments, the other for the
        # number of steps per environment. This only really matters when doing
        # training on a batch, but we've got to get the shapes to match here.
        if op.rnn_states_in is not None:
            if rnn_state is None:
                rnn_state = [self.rnn_zero_state()] * len(obs)
            policies, rnn_state = session.run(
                [op.policy, op.rnn_states_out],
                feed_dict={
                    op.states: [obs],
                    op.rnn_states_in: rnn_state
                })
        else:
            policies = session.run(op.policy, feed_dict={
                op.states: [obs]
            })
            rnn_state = [None] * len(obs)
        return policies[0], rnn_state

    @named_output(
        'states', 'actions', 'rewards', 'end_episode',
        'rnn_states', 'info')
    def run_agents(self, steps_per_env):
        """
        Create state/action sequences for each environment.

        This can be overridden by subclasses to use e.g. a replay buffer
        instead of sampling new states. The number of output environments
        doesn't have to equal the number of instantiated environments, although
        it does in this instantiation.

        Note that in addition to running the agents in the environment, this
        function also calls :method:`log_episode` function whenever an
        episode is complete, and it increments ``self.num_steps`` for each
        action taken.

        Parameters
        ----------
        steps_per_env : int

        Returns
        -------
        states : ndarray shape(steps_per_env+1, num_env, ...)
            There should be one more state than steps taken so as to include
            both the initial and final state.
        actions : ndarray shape(steps_per_env, num_env)
        rewards : ndarray shape(steps_per_env, num_env)
        end_episode : ndarray shape(steps_per_env, num_env), dtype bool
            True if the episode ended on that step, False otherwise.
        rnn_states : ndarray shape(num_env, ...)
            The initial internal state of the RNN for each environment at
            the beginning of each sequence. If an RNN isn't in use, this can
            be None or anything else.
        info : ndarray shape(steps_per_env, num_env)
            An array of info dictionaries for each environment.
        """
        num_env = len(self.envs)

        obs = []
        actions = []
        rewards = []
        end_episode = []
        initial_rnn_states = []
        infos = []
        rnn_zero_state = self.rnn_zero_state
        for env in self.envs:
            if not hasattr(env, '_ppo_last_obs'):
                env._ppo_last_obs = env.reset()
                env._ppo_rnn_state = rnn_zero_state
            obs.append(env._ppo_last_obs)
            initial_rnn_states.append(env._ppo_rnn_state)
        new_rnn_states = initial_rnn_states
        for _ in range(steps_per_env):
            policies, new_rnn_states = self.policy(obs[-num_env:], new_rnn_states)
            for env, policy, rnn_state in zip(
                    self.envs, policies, new_rnn_states):
                action = np.random.choice(len(policy), p=policy)
                new_obs, reward, done, info = env.step(action)
                if done:
                    self.num_episodes += 1
                    new_obs = env.reset()
                    rnn_state = rnn_zero_state
                env._ppo_last_obs = new_obs
                env._ppo_rnn_state = rnn_state
                obs.append(new_obs)
                actions.append(action)
                rewards.append(reward)
                end_episode.append(done)
                infos.append(info)
        self.num_steps += len(actions)

        out_shape = (steps_per_env, num_env)
        obs_shape = (steps_per_env+1, num_env) + obs[-1].shape
        return (
            np.array(obs).reshape(obs_shape),
            np.array(actions).reshape(out_shape),
            np.array(rewards).reshape(out_shape),
            np.array(end_episode).reshape(out_shape),
            np.array(initial_rnn_states),
            np.array(infos).reshape(out_shape),
        )

    @named_output('s', 'a', 'pi', 'r', 'G', 'A', 'v', 'm', 'c')
    def gen_training_batch(self, steps_per_env):
        """
        Create a batch of training data, including discounted rewards and
        advantages.
        """
        op = self.op
        session = self.session

        states, actions, rewards, end_episode, rnn_states, info = \
            self.run_agents(steps_per_env)
        # Note that there should be one more state than action/reward for
        # each environment.
        fd = {op.states: states}
        if op.rnn_states_in is not None:
            fd[op.rnn_states_in] = rnn_states
        policies, values = session.run([op.policy, op.v], feed_dict=fd)
        num_actions = policies.shape[-1]
        action_one_hot = np.eye(num_actions)[actions]
        action_prob = np.sum(policies[:-1] * action_one_hot, axis=-1)

        if self.reward_clip > 0:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)

        reward_mask = ~end_episode
        rnn_mask = np.roll(~end_episode, 1, axis=0)
        rnn_mask[0] = True

        gamma = self.gamma
        lmda = self.lmda * gamma
        advantages = rewards + gamma * reward_mask * values[1:] - values[:-1]
        returns = rewards.copy()
        returns[-1] += reward_mask[-1] * gamma * values[-1]
        for i in range(steps_per_env - 2, -1, -1):
            returns[i] += gamma * reward_mask[i] * returns[i+1]
            advantages[i] += lmda * reward_mask[i] * advantages[i+1]

        return (
            states[:-1], actions, action_prob, rewards, returns, advantages,
            values[:-1], rnn_mask, rnn_states
        )

    def train_batch(self, summarize=False):
        op = self.op
        session = self.session
        num_env = len(self.envs)
        env_idx = np.arange(num_env)
        assert num_env % self.envs_per_minibatch == 0

        batch = self.gen_training_batch(self.steps_per_env)

        for _ in range(self.epochs_per_batch):
            np.random.shuffle(env_idx)
            for idx in env_idx.reshape(-1, self.envs_per_minibatch):
                fd = {
                    op.states: batch.s[:,idx],
                    op.actions: batch.a[:,idx],
                    op.old_policy: batch.pi[:,idx],
                    op.old_value: batch.v[:,idx],
                    op.returns: batch.G[:,idx],
                    op.advantages: batch.A[:,idx],
                    op.rnn_mask: batch.m[:,idx],
                }
                if op.rnn_states_in is not None:
                    fd[op.rnn_states_in] = batch.c[idx]
                session.run(op.train, feed_dict=fd)

        if summarize:
            fd = {
                op.states: batch.s,
                op.actions: batch.a,
                op.old_policy: batch.pi,
                op.old_value: batch.v,
                op.returns: batch.G,
                op.advantages: batch.A,
                op.rnn_mask: batch.m,
            }
            if op.rnn_states_in is not None:
                fd[op.rnn_states_in] = batch.c
            summary = session.run(op.summary, feed_dict=fd)
            self.summary_writer.add_summary(summary, self.num_steps)

    def train(self, total_steps=None):
        last_report = last_save = self.num_steps - 1
        total_steps = total_steps or self.total_steps
        while self.num_steps < total_steps:
            summarize = last_report // self.report_every < self.num_steps // self.report_every
            self.train_batch(summarize=summarize)
            if last_save // self.save_every < self.num_steps // self.save_every:
                self.save_checkpoint()
                last_save = self.num_steps
        logger.info("FINISHED TRAINING")
