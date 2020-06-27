import logging
import numpy as np
from collections import defaultdict

import torch
import torch.optim as optim

from safelife.helper_utils import load_kwargs
from safelife.random import get_rng

from .utils import named_output, round_up
from .base_algo import BaseAlgo


logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()


class PPO(BaseAlgo):
    data_logger = None  # SafeLifeLogger instance

    num_steps = 0

    steps_per_env = 20
    num_minibatches = 4
    epochs_per_batch = 3

    gamma = 0.97
    lmda = 0.95
    learning_rate = 3e-4
    entropy_reg = 0.01
    entropy_clip = 1.0  # don't start regularization until it drops below this
    vf_coef = 0.5
    max_gradient_norm = 5.0
    eps_policy = 0.2  # PPO clipping for policy loss
    eps_value = 0.2  # PPO clipping for value loss
    rescale_policy_eps = False
    min_eps_rescale = 1e-3  # Only relevant if rescale_policy_eps = True
    reward_clip = 0.0
    policy_rectifier = 'relu'  # or 'elu' or ...more to come

    report_interval = 960
    test_interval = 100000

    compute_device = torch.device('cuda' if USE_CUDA else 'cpu')

    training_envs = None
    testing_envs = None

    checkpoint_attribs = ('model', 'optimizer', 'data_logger.cumulative_stats')

    def __init__(self, model, **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.model = model.to(self.compute_device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        self.load_checkpoint()

    @named_output('states actions rewards done policies values')
    def take_one_step(self, envs):
        states = [
            e.last_obs if hasattr(e, 'last_obs') else e.reset()
            for e in envs
        ]
        tensor_states = self.tensor(states, torch.float32)
        values, policies = self.model(tensor_states)
        values = values.detach().cpu().numpy()
        policies = policies.detach().cpu().numpy()
        actions = []
        rewards = []
        dones = []
        for policy, env in zip(policies, envs):
            action = get_rng().choice(len(policy), p=policy)
            obs, reward, done, info = env.step(action)
            if np.all(done):
                obs = env.reset()
            env.last_obs = obs
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        return states, actions, rewards, dones, policies, values

    @named_output('states actions action_prob returns advantages values')
    def gen_training_batch(self, steps_per_env, flat=True):
        """
        Run each environment a number of steps and calculate advantages.

        Parameters
        ----------
        steps_per_env : int
            Number of steps to take per environment.
        flat : bool
            If True, each output tensor will have shape
            ``(steps_per_env * num_env, ...)``.
            Otherwise, shape will be ``(steps_per_env, num_env, ...)``.
        """
        steps = [
            self.take_one_step(self.training_envs)
            for _ in range(steps_per_env)
        ]
        final_states = [e.last_obs for e in self.training_envs]
        tensor_states = self.tensor(final_states, torch.float32)
        final_vals = self.model(tensor_states)[0].detach().cpu().numpy()
        values = np.array([s.values for s in steps] + [final_vals])
        rewards = np.array([s.rewards for s in steps])
        done = np.array([s.done for s in steps])
        reward_mask = ~done

        # Calculate the discounted rewards
        gamma = self.gamma
        lmda = self.lmda
        returns = rewards.copy()
        returns[-1] += gamma * final_vals * reward_mask[-1]
        advantages = rewards + gamma * reward_mask * values[1:] - values[:-1]
        for i in range(steps_per_env - 2, -1, -1):
            returns[i] += gamma * reward_mask[i] * returns[i+1]
            advantages[i] += lmda * reward_mask[i] * advantages[i+1]

        # Calculate the probability of taking each selected action
        policies = np.array([s.policies for s in steps])
        actions = np.array([s.actions for s in steps])
        probs = np.take_along_axis(
            policies, actions[..., np.newaxis], axis=-1)[..., 0]

        def t(x, dtype=torch.float32):
            if flat:
                x = np.asanyarray(x)
                x = x.reshape(-1, *x.shape[2:])
            return torch.as_tensor(x, device=self.compute_device, dtype=dtype)

        self.num_steps += actions.size

        return (
            t([s.states for s in steps]), t(actions, torch.int64),
            t(probs), t(returns), t(advantages), t(values[:-1])
        )

    def calculate_loss(
            self, states, actions, old_policy, old_values, returns, advantages):
        """
        All parameters ought to be tensors on the appropriate compute device.
        """
        values, policy = self.model(states)
        a_policy = torch.gather(policy, -1, actions[..., np.newaxis])[..., 0]

        prob_diff = advantages.sign() * (1 - a_policy / old_policy)
        policy_loss = advantages.abs() * torch.clamp(prob_diff, min=-self.eps_policy)
        policy_loss = policy_loss.mean()

        v_clip = old_values + torch.clamp(
            values - old_values, min=-self.eps_value, max=+self.eps_value)
        value_loss = torch.max((v_clip - returns)**2, (values - returns)**2)
        value_loss = value_loss.mean()

        entropy = torch.sum(-policy * torch.log(policy + 1e-12), dim=-1)
        entropy_loss = torch.clamp(entropy.mean(), max=self.entropy_clip)
        entropy_loss *= -self.entropy_reg

        return entropy, policy_loss + value_loss * self.vf_coef + entropy_loss

    def train_batch(self, batch):
        num_samples = len(batch.states)
        idx = np.arange(num_samples)
        splits = np.linspace(
            0, num_samples, self.num_minibatches+2, dtype=int)[1:-1]

        for _ in range(self.epochs_per_batch):
            get_rng().shuffle(idx)
            for k in np.split(idx, splits):
                entropy, loss = self.calculate_loss(
                    batch.states[k], batch.actions[k], batch.action_prob[k],
                    batch.values[k], batch.returns[k], batch.advantages[k])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, steps):
        max_steps = self.num_steps + steps

        while self.num_steps < max_steps:
            next_report = round_up(self.num_steps, self.report_interval)
            next_test = round_up(self.num_steps, self.test_interval)

            batch = self.gen_training_batch(self.steps_per_env)
            self.train_batch(batch)

            self.save_checkpoint_if_needed()

            num_steps = self.num_steps

            if num_steps >= next_report and self.data_logger is not None:
                entropy, loss = self.calculate_loss(
                    batch.states, batch.actions, batch.action_prob,
                    batch.values, batch.returns, batch.advantages)
                loss = loss.item()
                entropy = entropy.mean().item()
                values = batch.values.mean().item()
                advantages = batch.advantages.mean().item()
                logger.info(
                    "n=%i: loss=%0.3g, entropy=%0.3f, val=%0.3g, adv=%0.3g",
                    num_steps, loss, entropy, values, advantages)
                self.data_logger.log_scalars({
                    "loss": loss,
                    "entropy": entropy,
                    "values": values,
                    "advantages": advantages,
                }, num_steps, 'ppo')

            if self.testing_envs and num_steps >= next_test:
                self.run_episodes(self.testing_envs)


class PPO_MultiAgent(PPO):
    """
    Similar the PPO class, but the environments are expected to return arrays
    of observations, rewards, and done flags for each agent.
    """
    @named_output('states actions rewards done obs active policies values agent_ids')
    def take_one_step(self, envs):
        """
        Take a single step in each of the environments.

        Note that when you have multiple agents, the number of output states
        (really, observations) won't match the number of environments. There
        won't even be the same number of observations per environment, since
        different environments can have different numbers of agents and some
        agents can leave an environment before others.

        Pseudo-code:

        for each env:
            get list of observations
            get list of active agents
        calculte all policies / actions
        split action for each env
        """
        states = []
        active = []
        env_indices = [0]
        for env in envs:
            if hasattr(env, 'last_obs'):
                obs = env.last_obs
                done = env.last_done
            else:
                obs = env.reset()
                done = np.tile(False, len(obs))
                env.num_resets = 0
            states.append(obs)
            active.append(~done)
            env_indices.append(env_indices[-1] + len(obs))

        states = np.concatenate(states)
        active = np.concatenate(active)

        tensor_states = self.tensor(states, torch.float32)
        values, policies = self.model(tensor_states)
        values = values.detach().cpu().numpy()
        policies = policies.detach().cpu().numpy()
        actions = [get_rng().choice(len(policy), p=policy) for policy in policies]

        rewards = []
        dones = []
        agent_id = []
        next_obs = []

        for i, env in enumerate(envs):
            k1, k2 = env_indices[i], env_indices[i+1]
            obs, reward, done, info = env.step(actions[k1:k2])
            next_obs += list(obs)
            rewards += list(reward)
            dones += list(done)
            agent_id += [(i, env.num_resets, j) for j in range(len(obs))]

            if np.all(done):
                obs = env.reset()
                done = np.tile(False, len(obs))
                env.num_resets += 1
            env.last_obs = obs
            env.last_done = done

        return (
            states, actions, rewards, dones,
            next_obs, active, policies, values, agent_id
        )

    @named_output('states actions action_prob returns advantages values')
    def gen_training_batch(self, steps_per_env):
        """
        Run each environment a number of steps and calculate advantages.

        Note that the output is flat, i.e., a single list of observations,
        actions, etc.

        ...should test to see how much slower this is than the old function.
        There are now a bunch of loops and dictionary lookups where we
        previously had numpy functions, but that's *probably* insignificant
        compared to the model evaluation.

        Parameters
        ----------
        steps_per_env : int
            Number of steps to take per environment.
        """
        assert steps_per_env > 0

        trajectories = defaultdict(lambda: {
            'states': [],
            'actions': [],
            'action_prob': [],
            'rewards': [],
            'values': [],
            'final_value': 0.0,
        })

        # Take a bunch of steps, and put them into trajectories associated with
        # each distinct agent
        for _ in range(steps_per_env):
            step = self.take_one_step(self.training_envs)
            for k, agent_id in enumerate(step.agent_ids):
                if step.active[k]:
                    t = trajectories[agent_id]
                    action = step.actions[k]
                    t['states'].append(step.states[k])
                    t['actions'].append(action)
                    t['action_prob'].append(step.policies[k, action])
                    t['rewards'].append(step.rewards[k])
                    t['values'].append(step.values[k])

        # For the final step in each environment, also calculate the value
        # function associated with the next observation
        tensor_states = self.tensor(step.obs, torch.float32)
        vals = self.model(tensor_states)[0].detach().cpu().numpy()
        for k, agent_id in enumerate(step.agent_ids):
            if not step.done[k]:
                t = trajectories[agent_id]
                t['final_value'] = vals[k]

        # Calculate the discounted rewards for each trajectory
        gamma = self.gamma
        lmda = self.lmda
        for t in trajectories.values():
            val0 = np.array(t['values'])
            val1 = np.append(t['values'][1:], t['final_value'])
            rewards = returns = np.array(t['rewards'])
            advantages = rewards + gamma * val1 - val0
            returns[-1] += gamma * t['final_value']
            for i in range(len(rewards) - 2, -1, -1):
                returns[i] += gamma * returns[i+1]
                advantages[i] += lmda * advantages[i+1]
            t['returns'] = returns
            t['advantages'] = advantages

        self.num_steps += steps_per_env * len(self.training_envs)

        def t(label, dtype=torch.float32):
            x = np.concatenate([d[label] for d in trajectories.values()])
            return torch.as_tensor(x, device=self.compute_device, dtype=dtype)

        return (
            t('states'), t('actions', torch.int64), t('action_prob'),
            t('returns'), t('advantages'), t('values')
        )
