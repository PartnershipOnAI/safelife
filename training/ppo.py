"""
Algorithm for Proximal Policy Optimization.
"""

import os
import logging
from types import SimpleNamespace
from collections import namedtuple
from functools import wraps

import numpy as np
import tensorflow as tf

from .wrappers import RewardsTracker

logger = logging.getLogger(__name__)

DEFAULT_LOGDIR = os.path.join(__file__, '../../data/tmp')
DEFAULT_LOGDIR = os.path.abspath(DEFAULT_LOGDIR)


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


class PPO(object):
    """
    Proximal policy optimization.

    Note that essentially all of these attributes can get overridden by
    subclasses, so the defaults set here are basically just for example.

    Attributes
    ----------
    gamma : ndarray
        Set of discount factors used to calculate the discounted rewards.
    lmda : float or ndarray
        Discount factor for generalized advantage estimator. If an array,
        it should be the same shape as gamma.
    policy_discount_weights : ndarray
        Relative importance of the advantages at the different discount
        factors in the policy loss function. Should sum to one.
    value_discount_weights : ndarray
        Relative importance of the advantages at the different discount
        factors in the value loss function. Should sum to one.
    vf_coef : float
        Overall coefficient of the value loss in the total loss function.
        Would be redundant with `value_discount_weights` if we didn't
        force that to sum to one.
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
    value_grad_rescaling : str
        One of [False, 'smooth', 'per_batch', 'per_state'].
        Sets the way in which value function is rescaled with entropy.
        This makes sure that the total gradient isn't dominated by the
        value function when entropy drops very low.
    policy_rectifier : str
        One of ['relu', 'elu'].
    """
    gamma = np.array([0.99], dtype=np.float32)
    lmda = 0.95  # generalized advantage estimation parameter
    policy_discount_weights = np.array([1.0], dtype=np.float32)
    value_discount_weights = np.array([1.0], dtype=np.float32)

    learning_rate = 1e-4
    entropy_reg = 0.01
    entropy_clip = 1.0  # don't start regularization until it drops below this
    vf_coef = 0.5
    max_gradient_norm = 5.0
    eps_clip = 0.2  # PPO clipping for both value and policy losses
    rescale_policy_eps = False
    min_eps_rescale = 1e-3  # Only relevant if rescale_policy_eps = True
    reward_clip = 0.0
    value_grad_rescaling = 'smooth'  # one of [False, 'smooth', 'per_batch', 'per_state']
    policy_rectifier = 'relu'  # or 'elu' or ...more to come

    steps_per_env = 20
    envs_per_minibatch = 4
    epochs_per_batch = 3
    total_steps = 5e6
    report_every = 5000
    save_every = 10000
    test_every = 100000

    def __init__(self, envs, logdir=DEFAULT_LOGDIR, saver_args={}, **kwargs):
        for key, val in kwargs.items():
            if (not key.startswith('_') and hasattr(self, key) and
                    not callable(getattr(self, key))):
                setattr(self, key, val)
            else:
                raise ValueError("Unrecognized parameter: '%s'" % (key,))

        self.envs = [RewardsTracker(env) for env in envs]

        self.op = SimpleNamespace()
        self.num_steps = 0
        self.num_episodes = 0
        self.build_graph()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.logger = tf.summary.FileWriter(logdir, self.session.graph)
        self.saver = tf.train.Saver(**saver_args)
        self.save_path = os.path.join(logdir, 'model')
        self.restore_checkpoint(logdir)

    def save_checkpoint(self):
        logger.info("Saving new checkpoint. %i episodes, %i steps.",
                    self.num_episodes, self.num_steps)
        self.op.num_steps.load(self.num_steps, self.session)
        self.op.num_episodes.load(self.num_episodes, self.session)
        self.saver.save(self.session, self.save_path, self.num_steps)

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
            return
        with open(checkpoint_path) as checkpoint_file:
            line = checkpoint_file.readline()
        match = re.match(r'.*"(.+)"', line)
        if not match:
            return
        last_checkpoint = os.path.split(match.group(1))[1]
        last_checkpoint = os.path.join(logdir, last_checkpoint)
        try:
            self.saver.restore(self.session, last_checkpoint)
        except ValueError:
            if raise_on_error:
                raise
            else:
                return
        self.num_steps, self.num_episodes = self.session.run(
            [self.op.num_steps, self.op.num_episodes])
        logger.info("Restoring old checkpoint. %i episodes, %i steps.",
                    self.num_episodes, self.num_steps)

    def build_graph(self):
        op = self.op
        input_space = self.envs[0].observation_space
        n_gamma = len(self.gamma)
        op.states = tf.placeholder(input_space.dtype, [None, None] + list(input_space.shape), name="state")
        op.actions = tf.placeholder(tf.int32, [None, None], name="actions")
        op.old_policy = tf.placeholder(tf.float32, [None, None], name="old_policy")
        op.returns = tf.placeholder(tf.float32, [None, None, n_gamma], name="returns")
        op.advantages = tf.placeholder(tf.float32, [None, None, n_gamma], name="advantages")
        op.old_value = tf.placeholder(tf.float32, [None, None, n_gamma], name="old_value")
        op.learning_rate = tf.constant(self.learning_rate, name="learning_rate")
        op.eps_clip = tf.constant(self.eps_clip, name="eps_clip")
        op.rnn_mask = tf.fill(tf.shape(op.states)[:2], True, name="rnn_mask")
        op.policy_discount_weights = tf.constant(self.policy_discount_weights, name="policy_discount_weights")
        op.value_discount_weights = tf.constant(self.value_discount_weights, name="value_discount_weights")
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
            prob_diff = tf.sign(op.advantages) * (1 - a_policy / op.old_policy)[..., None]
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
            policy_loss = tf.reduce_mean(policy_loss * op.policy_discount_weights)
        with tf.name_scope("entropy"):
            op.entropy = tf.reduce_sum(
                -op.policy * tf.log(op.policy + 1e-12), axis=-1)
            mean_entropy = tf.reduce_mean(op.entropy)
            pseudo_entropy = tf.stop_gradient(
                tf.reduce_sum(op.policy*(1-op.policy), axis=-1))
            avg_pseudo_entropy = tf.reduce_mean(pseudo_entropy)
            smoothed_pseudo_entropy = tf.get_variable(
                'smoothed_pseudo_entropy', initializer=tf.constant(1.0))
            # The first term in the entropy loss encourages higher entropy
            # in the policy, encouraging exploration.
            # Note that this uses the pseudo-entropy rather than the
            # conventional entropy. This is because the derivative of the
            # normal entropy diverges at zero.
            entropy_loss = -self.entropy_reg * tf.minimum(avg_pseudo_entropy, self.entropy_clip)
            # The second term in the entropy loss is just used to adjust the
            # smoothed pseudo entropy.
            entropy_loss += 0.5 * tf.square(avg_pseudo_entropy - smoothed_pseudo_entropy)
        with tf.name_scope("value_loss"):
            v_clip = op.old_value + tf.clip_by_value(
                op.v - op.old_value, -op.eps_clip, op.eps_clip)
            value_loss = tf.maximum(
                tf.square(op.v - op.returns), tf.square(v_clip - op.returns))
            # Rescale the value function with entropy.
            # The gradient of the policy function becomes very small when
            # the entropy is very low, essentially because it means the softmax
            # of the policy logits is being saturated. By rescaling the value
            # loss we attempt to make it have the same relative importance
            # as the policy loss. Not clear how necessary this is.
            if self.value_grad_rescaling == 'per_state':
                value_loss *= pseudo_entropy
            elif self.value_grad_rescaling == 'per_batch':
                value_loss *= avg_pseudo_entropy
            elif self.value_grad_rescaling == 'smooth':
                value_loss *= tf.stop_gradient(smoothed_pseudo_entropy)
            elif self.value_grad_rescaling:
                raise ValueError("Unrecognized value reweighting type: '%s'" % (
                    self.value_grad_rescaling,))
            value_loss = 0.5 * tf.reduce_mean(value_loss * op.value_discount_weights)

        with tf.name_scope("trainer"):
            total_loss = policy_loss + value_loss * self.vf_coef + entropy_loss
            optimizer = self.build_optimizer(op.learning_rate)
            grads, variables = zip(*optimizer.compute_gradients(total_loss))
            op.grads = grads
            if self.max_gradient_norm > 0:
                grads2, _ = tf.clip_by_global_norm(grads, self.max_gradient_norm)
            op.train = optimizer.apply_gradients(zip(grads2, variables))

        with tf.name_scope("rollouts"):
            for i in range(n_gamma):
                k = str(i+1)
                tf.summary.scalar("returns_"+k, tf.reduce_mean(op.returns[...,i]))
                tf.summary.scalar("advantages_"+k, tf.reduce_mean(op.advantages[...,i]))
                tf.summary.scalar("values_"+k, tf.reduce_mean(op.v[...,i]))
                tf.summary.histogram("returns_"+k, op.returns[...,i])
                tf.summary.histogram("advantages_"+k, op.advantages[...,i])
                tf.summary.histogram("values_"+k, op.v[...,i])
        tf.summary.scalar("entropy", mean_entropy)
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

        There should be an equal number of logits and possible actions,
        and the number of value functions should match the number of distinct
        discount factors (gamma).

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

    @named_output(
        'states', 'actions', 'rewards', 'end_episode', 'times_up',
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
        times_up : ndarray shape(steps_per_env, num_env), dtype bool
            True if the episode ended on that step due to the time limit being
            exceeded, False otherwise.
        rnn_states : ndarray shape(num_env, ...)
            The initial internal state of the RNN for each environment at
            the beginning of each sequence. If an RNN isn't in use, this can
            be None or anything else.
        info : ndarray shape(steps_per_env, num_env)
            An array of info dictionaries for each environment.
        """
        op = self.op
        session = self.session
        num_env = len(self.envs)

        obs = []
        actions = []
        rewards = []
        end_episode = []
        times_up = []
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
            if op.rnn_states_in is not None:
                policies, new_rnn_states = session.run(
                    [op.policy, op.rnn_states_out],
                    feed_dict={
                        op.states: [obs[-num_env:]],
                        op.rnn_states_in: new_rnn_states
                    })
            else:
                policies = session.run(op.policy, feed_dict={
                    op.states: [obs[-num_env:]]
                })
            for env, policy, rnn_state in zip(
                    self.envs, policies[0], new_rnn_states):
                action = np.random.choice(len(policy), p=policy)
                new_obs, reward, done, info = env.step(action)
                if done:
                    self.log_episode(env)
                    new_obs = env.reset()
                    rnn_state = rnn_zero_state
                env._ppo_last_obs = new_obs
                env._ppo_rnn_state = rnn_state
                obs.append(new_obs)
                actions.append(action)
                rewards.append(reward)
                end_episode.append(done)
                times_up.append(info.get('times_up', done))
                infos.append(info)
        self.num_steps += len(actions)

        out_shape = (steps_per_env, num_env)
        obs_shape = (steps_per_env+1, num_env) + obs[-1].shape
        return (
            np.array(obs).reshape(obs_shape),
            np.array(actions).reshape(out_shape),
            np.array(rewards).reshape(out_shape),
            np.array(end_episode).reshape(out_shape),
            np.array(times_up).reshape(out_shape),
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

        states, actions, rewards, end_episode, times_up, rnn_states, info = \
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

        reward_mask = ~times_up[..., np.newaxis]
        rnn_mask = np.roll(~end_episode, 1, axis=0)
        rnn_mask[0] = True
        rewards = rewards[..., np.newaxis]

        gamma = self.gamma
        lmda = self.lmda * gamma
        n_gamma = len(gamma)
        advantages = rewards + gamma * reward_mask * values[1:] - values[:-1]
        returns = np.broadcast_to(rewards, rewards.shape[:-1] + (n_gamma,)).copy()
        returns[-1] += reward_mask[-1] * gamma * values[-1]
        for i in range(steps_per_env - 2, -1, -1):
            returns[i] += gamma * reward_mask[i] * returns[i+1]
            advantages[i] += lmda * reward_mask[i] * advantages[i+1]

        return (
            states[:-1], actions, action_prob, rewards[...,0], returns, advantages,
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
            self.logger.add_summary(summary, self.num_steps)

    def log_episode(self, env):
        self.num_episodes += 1
        summary = tf.Summary()
        for key, val in env.episode_info.items():
            summary.value.add(tag='episode/'+key, simple_value=val)
        summary.value.add(tag='episode/completed', simple_value=self.num_episodes)
        self.logger.add_summary(summary, self.num_steps)
        logger.info(
            "Episode %i: length=%i, reward=%0.1f",
            self.num_episodes, env.episode_info['length'], env.episode_info['reward'])

    def train(self, total_steps=None):
        last_report = last_save = last_test = self.num_steps - 1
        total_steps = total_steps or self.total_steps
        while self.num_steps < total_steps:
            summarize = last_report // self.report_every < self.num_steps // self.report_every
            self.train_batch(summarize=summarize)
            if last_save // self.save_every < self.num_steps // self.save_every:
                self.save_checkpoint()
                last_save = self.num_steps
            if self.test_every and last_test // self.test_every < self.num_steps // self.test_every:
                self.run_safety_test()
                last_test = self.num_steps
        logger.info("FINISHED TRAINING")

    def run_safety_test(self):
        """
        To be implemented by subclasses.
        """
        pass
