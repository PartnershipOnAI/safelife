import logging
import numpy as np
from scipy.interpolate import UnivariateSpline

import torch
import torch.optim as optim

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
        return zip(*data)

    def __len__(self):
        return min(self.idx, self.capacity)


class MultistepReplayBuffer(object):
    def __init__(self, capacity, num_env, n_step, gamma):
        self.capacity = capacity
        self.idx = 0
        self.states = np.zeros(capacity, dtype=object)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        self.num_env = num_env
        self.n_step = n_step
        self.gamma = gamma
        self.tail_length = n_step * num_env

    def push(self, state, action, reward, done):
        idx = self.idx % self.capacity
        self.idx += 1
        self.states[idx] = state
        self.actions[idx] = action
        self.done[idx] = done
        self.rewards[idx] = reward

        # Now discount the reward and add to prior rewards
        n = np.arange(1, self.n_step)
        idx_prior = idx - n * self.num_env
        prior_done = np.cumsum(self.done[idx_prior]) > 0
        gamma = self.gamma**(n) * ~prior_done
        self.rewards[idx_prior] += gamma * reward
        self.done[idx_prior] = prior_done | done

    @named_output("state action reward next_state done")
    def sample(self, batch_size):
        assert self.idx >= batch_size + self.tail_length

        idx = self.idx % self.capacity
        i1 = idx - 1 - get_rng().choice(len(self), batch_size, replace=False)
        i0 = i1 - self.tail_length

        return (
            list(self.states[i0]),  # don't want dtype=object in output
            self.actions[i0],
            self.rewards[i0],
            list(self.states[i1]),  # states n steps later
            self.done[i0],  # whether or not the episode ended before n steps
        )

    def __len__(self):
        return max(min(self.idx, self.capacity) - self.tail_length, 0)


class DQN(BaseAlgo):
    data_logger = None

    num_steps = 0

    gamma = 0.97
    multi_step_learning = 15
    training_batch_size = 96
    optimize_interval = 32
    learning_rate = 3e-4
    epsilon_schedule = UnivariateSpline(  # Piecewise linear schedule
        [5e4, 5e5, 4e6],
        [1, 0.1, 0.03], s=0, k=1, ext='const')
    epsilon_testing = 0.01

    replay_initial = 40000
    replay_size = 100000
    target_update_interval = 10000

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
        self.replay_buffer = MultistepReplayBuffer(
            self.replay_size, len(self.training_envs),
            self.multi_step_learning, self.gamma)

        self.load_checkpoint()
        self.epsilon = self.epsilon_schedule(self.num_steps)

    def update_target(self):
        self.target_model.load_state_dict(self.training_model.state_dict())

    @named_output('states actions rewards done qvals')
    def take_one_step(self, envs, add_to_replay=False):
        states = [
            e.last_state if hasattr(e, 'last_state') else e.reset()
            for e in self.training_envs
        ]
        tensor_states = torch.tensor(states, device=self.compute_device, dtype=torch.float32)
        qvals = self.training_model(tensor_states).detach().cpu().numpy()

        num_states, num_actions = qvals.shape
        actions = np.argmax(qvals, axis=-1)
        random_actions = get_rng().integers(num_actions, size=num_states)
        use_random = get_rng().random(num_states) < self.epsilon
        actions = np.choose(use_random, [actions, random_actions])
        rewards = []
        dones = []

        for env, state, action in zip(self.training_envs, states, actions):
            next_state, reward, done, info = env.step(action)
            if done:
                next_state = env.reset()
            env.last_state = next_state
            if add_to_replay:
                self.replay_buffer.push(state, action, reward, done)
                self.num_steps += 1
            rewards.append(reward)
            dones.append(done)

        return states, actions, rewards, dones, qvals

    def optimize(self, report=False):
        if len(self.replay_buffer) < self.replay_initial:
            return

        state, action, reward, next_state, done = \
            self.replay_buffer.sample(self.training_batch_size)

        state = torch.tensor(state, device=self.compute_device, dtype=torch.float32)
        next_state = torch.tensor(next_state, device=self.compute_device, dtype=torch.float32)
        action = torch.tensor(action, device=self.compute_device, dtype=torch.int64)
        reward = torch.tensor(reward, device=self.compute_device, dtype=torch.float32)
        done = torch.tensor(done, device=self.compute_device, dtype=torch.float32)

        q_values = self.training_model(state)
        next_q_values = self.target_model(next_state).detach()

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

            self.epsilon = self.epsilon_schedule(self.num_steps)
            self.take_one_step(self.training_envs, add_to_replay=True)

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
