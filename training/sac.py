import logging
from collections import defaultdict
from itertools import chain

import torch
import numpy as np

from safelife.helper_utils import load_kwargs
from safelife.random import get_rng

from .base_algo import BaseAlgo
from .utils import named_output, round_up


logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.idx = 0
        self.buffer = np.zeros(capacity, dtype=object)

    def push(self, *data):
        self.buffer[self.idx % self.capacity] = data
        self.idx += 1

    def sample(self, batch_size):
        sub_buffer = self.buffer[:self.idx]
        data = get_rng().choice(sub_buffer, batch_size, replace=False)
        return list(zip(*data))

    def __len__(self):
        return min(self.idx, self.capacity)


def entropy_k(pi, k, offset=False):
    if k == 0:
        y = -pi * torch.log(pi)
    elif k >= -1 and offset:
        y = -pi * (pi**k - 1) / k
    elif offset:
        y = pi * (pi**k - 1) / k
    elif k >= -1:
        y = -pi**(k+1) / k
    else:
        y = pi**(k+1) / k
    return torch.sum(y, axis=-1)


class SAC(BaseAlgo):
    data_logger = None

    num_steps = 0

    gamma = 0.97
    entropy_coef = 0.001
    k_entropy = -2
    polyak = 0.995

    multi_step_learning = 1
    training_batch_size = 96
    optimize_interval = 32
    learning_rate = 3e-4

    replay_initial = 40000
    replay_size = 100000
    target_update_interval = 10000

    report_interval = 256
    test_interval = 100000

    compute_device = torch.device('cuda' if USE_CUDA else 'cpu')

    training_envs = None
    testing_envs = None

    checkpoint_attribs = (
        'qnet1', 'qnet2', 'qnet1_t', 'qnet2_t', 'policy', 'optimizer',
        'data_logger.cumulative_stats',
    )

    def __init__(self, policy_network, q_network, **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.qnet1 = q_network().to(self.compute_device)
        self.qnet2 = q_network().to(self.compute_device)
        self.qnet1_t = q_network().to(self.compute_device)
        self.qnet2_t = q_network().to(self.compute_device)
        self.policy = policy_network.to(self.compute_device)

        self.qnet1_t.load_state_dict(self.qnet1.state_dict())
        self.qnet2_t.load_state_dict(self.qnet2.state_dict())

        self.optimizer = torch.optim.Adam(
            chain(
                self.qnet1.parameters(),
                self.qnet2.parameters(),
                self.policy.parameters()
            ), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(self.replay_size)
        self.agent_trajectories = defaultdict(lambda: np.empty(
            self.multi_step_learning,
            dtype=[('obs', object), ('action', int), ('reward', float)])
        )

        self.load_checkpoint()

    @named_output('obs actions rewards done next_obs agent_ids')
    def take_one_step(self, envs):
        obs, agent_ids = self.obs_for_envs(envs)

        rng = get_rng()
        if envs and self.num_steps < self.replay_initial:
            actions = rng.integers(envs[0].action_space.n, size=len(obs))
        else:
            # Maybe do something different for testing?
            tensor_obs = self.tensor(obs, torch.float32)
            policies = self.policy(tensor_obs).detach().cpu().numpy()
            actions = [rng.choice(len(policy), p=policy) for policy in policies]

        next_obs, rewards, done = self.act_on_envs(envs, actions)

        return obs, actions, rewards, done, next_obs, agent_ids

    def add_to_replay(self, step):
        """
        Add a step from 'take_one_step' to the replay buffer.

        Note that this uses multi-step learning in general, although if
        multi_step_learning = 1 it reduces to a much simpler algorithm.
        """
        gamma = self.gamma**np.arange(1, self.multi_step_learning)
        for obs, act, reward, done, next_obs, agent_id in zip(*step):
            trajectory = self.agent_trajectories[agent_id]
            # shift the trajectory forward
            obs0, act0, reward0 = trajectory[-1]
            trajectory[1:] = trajectory[:-1]
            trajectory[0] = obs, act, reward
            # Calculated discounted reward.
            trajectory['reward'][1:] += reward * gamma
            if obs0 is not None:
                self.replay_buffer.push(obs0, act0, reward0, obs, done)
            if done:
                # Add the rest of the trajectory to replay buffer, since
                # the state is terminal and no other discounted rewards
                # get added.
                for obs, act, reward in trajectory:
                    if obs is None:
                        break
                    self.replay_buffer.push(obs, act, reward, next_obs, done)
                # Remove the trajectory to free up memory.
                del self.agent_trajectories[agent_id]

    def optimize(self, report=False):
        if len(self.replay_buffer) < self.replay_initial:
            return

        obs, action, reward, next_obs, done = \
            self.replay_buffer.sample(self.training_batch_size)

        obs = self.tensor(obs, torch.float32)
        next_obs = self.tensor(next_obs, torch.float32)
        action = self.tensor(action, torch.int64)
        reward = self.tensor(reward, torch.float32)
        done = self.tensor(done, torch.float32)

        # Calculate the target for the q functions
        q1_t = self.qnet1(next_obs).detach()
        q2_t = self.qnet2(next_obs).detach()
        qmin_t = torch.min(q1_t, q2_t)
        pi_t = self.policy(next_obs).detach()
        V_t = torch.sum(pi_t * qmin_t, axis=-1)
        entropy_t = entropy_k(pi_t, self.k_entropy)
        discount = self.gamma**self.multi_step_learning * (1 - done)
        target = reward + discount * (V_t + self.entropy_coef * entropy_t)

        # MSE loss for the q functions
        q1 = self.qnet1(obs)
        q2 = self.qnet2(obs)
        q1_act = q1.gather(1, action.unsqueeze(1)).squeeze(1)
        q2_act = q2.gather(1, action.unsqueeze(1)).squeeze(1)
        q1_loss = torch.mean((q1_act - target)**2)
        q2_loss = torch.mean((q2_act - target)**2)

        # For the policy, do gradient _ascent_ (hence the final minus sign)
        # on the value function.
        qmin = torch.min(q1, q2).detach()
        pi = self.policy(obs)
        V_mean = torch.mean(torch.sum(pi * qmin, axis=-1))
        entropy = torch.mean(entropy_k(pi, self.k_entropy))
        pi_loss = -(V_mean + self.entropy_coef * entropy)

        # Run the optimizer. Not that each of the losses is independent of
        # the others (no parameter sharing)
        self.optimizer.zero_grad()
        q1_loss.backward()
        q2_loss.backward()
        pi_loss.backward()
        self.optimizer.step()

        # Update target weights with polyak averaging
        params = chain(self.qnet1.parameters(), self.qnet2.parameters())
        params_t = chain(self.qnet1_t.parameters(), self.qnet2_t.parameters())
        with torch.no_grad():
            for p, p_targ in zip(params, params_t):
                # NB: We use an in-place operations "mul_", "add_" to update
                # target params, as opposed to "mul" and "add", which would
                # make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # Finally, report some stats
        if report and self.data_logger is not None:
            data = {
                "q1_loss": q1_loss.item(),
                "q2_loss": q2_loss.item(),
                "entropy": torch.mean(entropy_k(pi, 0)).item(),
                "value_func": V_mean.item(),
            }
            logger.info(
                "n=%i: val=%0.3g, entropy=%0.3g", self.num_steps,
                data['value_func'], data['entropy'])
            self.data_logger.log_scalars(data, self.num_steps, 'sac')

    def train(self, steps):
        needs_report = True
        max_steps = self.num_steps + steps

        while self.num_steps < max_steps:
            self.is_training = False
            num_steps = self.num_steps
            next_opt = round_up(num_steps, self.optimize_interval)
            next_report = round_up(num_steps, self.report_interval)
            next_test = round_up(num_steps, self.test_interval)

            step_data = self.take_one_step(self.training_envs)
            self.add_to_replay(step_data)
            self.num_steps += len(self.training_envs)

            if len(self.replay_buffer) < self.replay_initial:
                continue

            if self.num_steps >= next_report:
                needs_report = True

            if self.num_steps >= next_opt:
                self.optimize(needs_report)
                needs_report = False

            self.save_checkpoint_if_needed()

            if self.testing_envs and self.num_steps >= next_test:
                self.is_training = True
                self.run_episodes(self.testing_envs)
