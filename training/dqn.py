import logging
from collections import defaultdict
import numpy as np
from scipy.interpolate import UnivariateSpline

import torch
import torch.optim as optim

from safelife.helper_utils import load_kwargs
from safelife.random import get_rng

from .base_algo import BaseAlgo, HyperParam
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


class DQN(BaseAlgo):
    data_logger = None

    num_steps = 0

    gamma: HyperParam = 0.97
    multi_step_learning: HyperParam = 5
    training_batch_size: HyperParam = 96
    optimize_interval: HyperParam = 32
    learning_rate: HyperParam = 3e-4
    epsilon_schedule = UnivariateSpline(  # Piecewise linear schedule
        [5e4, 5e5, 4e6],
        [1, 0.5, 0.03], s=0, k=1, ext='const')
    epsilon_testing = 0.01

    replay_initial: HyperParam = 40000
    replay_size: HyperParam = 100000
    target_update_interval: HyperParam = 10000

    report_interval = 256
    test_interval = 100000

    compute_device = torch.device('cuda' if USE_CUDA else 'cpu')

    training_envs = None
    testing_envs = None

    checkpoint_attribs = (
        'training_model', 'target_model', 'optimizer',
        'data_logger.cumulative_stats',
    )

    def __init__(self, training_model, target_model, **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.training_model = training_model.to(self.compute_device)
        self.target_model = target_model.to(self.compute_device)
        self.optimizer = optim.Adam(
            self.training_model.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.replay_size)
        self.agent_trajectories = defaultdict(lambda: np.empty(
            self.multi_step_learning,
            dtype=[('obs', object), ('action', int), ('reward', float)])
        )

        self.load_checkpoint()
        self.epsilon = self.epsilon_schedule(self.num_steps)

        if self.data_logger is not None:
            self.data_logger.save_hyperparameters({'dqn': self.hyperparams})

    def update_target(self):
        self.target_model.load_state_dict(self.training_model.state_dict())

    @named_output('obs actions rewards done next_obs agent_ids')
    def take_one_step(self, envs):
        obs, agent_ids = self.obs_for_envs(envs)

        obs_tensor = self.tensor(obs, torch.float32)
        qvals = self.training_model(obs_tensor).detach().cpu().numpy()

        num_states, num_actions = qvals.shape
        actions = np.argmax(qvals, axis=-1)
        random_actions = get_rng().integers(num_actions, size=num_states)
        use_random = get_rng().random(num_states) < self.epsilon
        actions = np.choose(use_random, [actions, random_actions])

        next_obs, rewards, done = self.act_on_envs(envs, actions)

        return obs, actions, rewards, done, next_obs, agent_ids

    def add_to_replay(self, step):
        """
        Add a step from 'take_one_step' to the replay buffer.
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

        q_values = self.training_model(obs)
        next_q_values = self.target_model(next_obs).detach()

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value, next_action = next_q_values.max(1)
        discount = self.gamma**self.multi_step_learning * (1 - done)
        expected_q_value = reward + discount * next_q_value

        loss = torch.mean((q_value - expected_q_value)**2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if report and self.data_logger is not None:
            data = {
                "loss": loss.item(),
                "epsilon": self.epsilon,
                "q_model_mean": q_values.mean().item(),
                "q_model_max": q_values.max(1)[0].mean().item(),
                "q_target_mean": next_q_values.mean().item(),
                "q_target_max": next_q_value.mean().item(),
            }
            logger.info(
                "n=%i: loss=%0.3g, q_mean=%0.3g, q_max=%0.3g", self.num_steps,
                data['loss'], data['q_model_mean'], data['q_model_max'])
            self.data_logger.log_scalars(data, self.num_steps, 'dqn')

    def train(self, steps):
        needs_report = True
        max_steps = self.num_steps + steps

        while self.num_steps < max_steps:
            num_steps = self.num_steps
            next_opt = round_up(num_steps, self.optimize_interval)
            next_update = round_up(num_steps, self.target_update_interval)
            next_report = round_up(num_steps, self.report_interval)
            next_test = round_up(num_steps, self.test_interval)

            self.epsilon = float(self.epsilon_schedule(self.num_steps))
            step_data = self.take_one_step(self.training_envs)
            self.add_to_replay(step_data)
            self.num_steps += len(self.training_envs)

            num_steps = self.num_steps

            if len(self.replay_buffer) < self.replay_initial:
                continue

            if num_steps >= next_report:
                needs_report = True

            if num_steps >= next_opt:
                self.optimize(needs_report)
                needs_report = False

            if num_steps >= next_update:
                self.target_model.load_state_dict(self.training_model.state_dict())

            self.save_checkpoint_if_needed()

            if self.testing_envs and num_steps >= next_test:
                self.epsilon = self.epsilon_testing
                self.run_episodes(self.testing_envs)
