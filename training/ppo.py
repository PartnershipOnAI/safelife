import logging
import numpy as np
from collections import defaultdict

import torch
import torch.optim as optim

from safelife.helper_utils import load_kwargs
from safelife.random import get_rng

from .utils import named_output, round_up, get_compute_device, recursive_shape
from .base_algo import BaseAlgo

try:
    import torch_xla.core.xla_model as xm
    import torch_xla
    assert torch_xla
except ImportError:
    pass  # no TPU

logger = logging.getLogger(__name__)


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

    compute_device = get_compute_device()

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

    @named_output('obs actions rewards done next_obs agent_ids policies values')
    def take_one_step(self, envs):
        obs, agent_ids = self.obs_for_envs(envs)

        tensor_obs = self.tensor(obs, torch.float32)
        values, policies = self.model_forward(tensor_obs)
        values = values.detach().cpu().numpy()
        policies = policies.detach().cpu().numpy()
        if self.compute_device.type == "xla":
            # correct after low precision #floatlife
            policies = policies.astype("float16")
        actions = []
        for policy in policies:
            try:
                actions.append(get_rng().choice(len(policy), p=policy))
            except ValueError:
                print("Logits:", policy, "sum to", np.sum(policy))
                raise

        next_obs, rewards, done = self.act_on_envs(envs, actions)

        return obs, actions, rewards, done, next_obs, agent_ids, policies, values

    @named_output('obs actions action_prob returns advantages values')
    def gen_training_batch(self, steps_per_env):
        """
        Run each environment a number of steps and calculate advantages.

        Note that the output is flat, i.e., a single list of observations,
        actions, etc.

        Parameters
        ----------
        steps_per_env : int
            Number of steps to take per environment.
        """
        assert steps_per_env > 0

        trajectories = defaultdict(lambda: {
            'obs': [],
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
                t = trajectories[agent_id]
                action = step.actions[k]
                t['obs'].append(step.obs[k])
                t['actions'].append(action)
                t['action_prob'].append(step.policies[k, action])
                t['rewards'].append(step.rewards[k])
                t['values'].append(step.values[k])

        # For the final step in each environment, also calculate the value
        # function associated with the next observation
        tensor_obs = self.tensor(step.next_obs, torch.float32)
        vals = self.model_forward(tensor_obs)[0].detach().cpu().numpy()
        for k, agent_id in enumerate(step.agent_ids):
            if not step.done[k]:
                trajectories[agent_id]['final_value'] = vals[k]

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
            t('obs'), t('actions', torch.int64), t('action_prob'),
            t('returns'), t('advantages'), t('values')
        )

    def model_forward(self, obs):
        "Here to be overriden if state is required."
        return self.model(obs)

    def calculate_loss(
            self, obs, actions, old_policy, old_values, returns, advantages):
        """
        All parameters ought to be tensors on the appropriate compute device.
        """
        values, policy = self.model_forward(obs)
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
        num_samples = len(batch.obs)
        idx = np.arange(num_samples)
        splits = np.linspace(
            0, num_samples, self.num_minibatches+2, dtype=int)[1:-1]

        for _ in range(self.epochs_per_batch):
            get_rng().shuffle(idx)
            for k in np.split(idx, splits):
                entropy, loss = self.calculate_loss(
                    batch.obs[k], batch.actions[k], batch.action_prob[k],
                    batch.values[k], batch.returns[k], batch.advantages[k])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.compute_device.type == "xla":
                    xm.mark_step()

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
                    batch.obs, batch.actions, batch.action_prob,
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

class LSTM_PPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # (hidden state, cell state) <--- XXX need better init?
        #self.lstm_state = (torch.randn(1, 16, 576), torch.randn(1, 16, 576))
        self.default_state = (torch.zeros(1, 16, 576), torch.zeros(1, 16, 576))

    def model_forward(self, obs):
        ret = self.model(obs, self.lstm_state)
        result = ret[:-1]  # trim off & discard LSTM state
                           # on the speculative theory it needn't be kept except
                           # for forward passes from take_one_step
        return ret

    @named_output('obs actions rewards done next_obs agent_ids policies values')
    def take_one_step(self, envs):
        obs, agent_ids = self.obs_for_envs(envs)
        state = [getattr(e, "lstm_state", self.default_state) for e in envs]

        tensor_obs = self.tensor(obs, torch.float32)
        values, policies, new_state = self.model(tensor_obs, self.lstm_state)
        for i, e in enumerate(envs):
            e.lstm_state = new_state[i]
        values = values.detach().cpu().numpy()
        policies = policies.detach().cpu().numpy()
        if self.compute_device.type == "xla":
            # correct after low precision #floatlife
            policies = policies.astype("float16")
        actions = []
        for policy in policies:
            try:
                actions.append(get_rng().choice(len(policy), p=policy))
            except ValueError:
                print("Logits:", policy, "sum to", np.sum(policy))
                raise

        next_obs, rewards, done = self.act_on_envs(envs, actions)

        return obs, actions, rewards, done, next_obs, agent_ids, policies, values, state

    @named_output('obs actions action_prob returns advantages values')
    def gen_training_batch(self, steps_per_env):
        """
        Run each environment a number of steps and calculate advantages.

        Note that the output is flat, i.e., a single list of observations,
        actions, etc.

        Parameters
        ----------
        steps_per_env : int
            Number of steps to take per environment.
        """
        assert steps_per_env > 0

        trajectories = defaultdict(lambda: {
            'obs': [],
            'actions': [],
            'action_prob': [],
            'rewards': [],
            'values': [],
            'lstm_state': [],
            'ongoing': [],
            'final_value': 0.0,
        })

        # Take a bunch of steps, and put them into trajectories associated with
        # each distinct agent
        for _ in range(steps_per_env):
            with torch.no_grad():
                step = self.take_one_step(self.training_envs)
            for k, agent_id in enumerate(step.agent_ids):
                t = trajectories[agent_id]
                action = step.actions[k]
                t['obs'].append(step.obs[k])
                t['actions'].append(action)
                t['action_prob'].append(step.policies[k, action])
                t['rewards'].append(step.rewards[k])
                t['values'].append(step.values[k])
                t['ongoing'].append(not step.done[k])

        # For the final step in each environment, also calculate the value
        # function associated with the next observation
        tensor_obs = self.tensor(step.next_obs, torch.float32)
        vals = self.model_forward(tensor_obs)[0].detach().cpu().numpy()
        for k, agent_id in enumerate(step.agent_ids):
            if not step.done[k]:
                trajectories[agent_id]['final_value'] = vals[k]

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
            t('obs'), t('actions', torch.int64), t('action_prob'),
            t('returns'), t('advantages'), t('values'), t('lstm_state'), t('ongoing')
        )

