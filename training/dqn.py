import numpy as np

import torch
import torch.optim as optim

from safelife.helper_utils import load_kwargs

from . import checkpointing
from .utils import round_up


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
        data = np.random.choice(sub_buffer, batch_size, replace=False)
        return zip(*data)

    def __len__(self):
        return min(self.idx, self.capacity)


class DQN(object):
    summary_writer = None
    logdir = None

    num_steps = 0
    num_episodes = 0

    gamma = 0.97
    training_batch_size = 64
    optimize_freq = 16
    learning_rate = 3e-4

    replay_initial = 40000
    replay_size = 100000
    target_update_freq = 10000

    checkpoint_freq = 100000
    num_checkpoints = 3
    report_freq = 256
    test_freq = 100000

    compute_device = torch.device('cuda' if USE_CUDA else 'cpu')

    training_envs = None
    testing_envs = None

    epsilon = 0.0  # for exploration

    def __init__(self, training_model, target_model, **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.training_model = training_model.to(self.compute_device)
        self.target_model = target_model.to(self.compute_device)
        self.optimizer = optim.Adam(
            self.training_model.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.replay_size)

        checkpointing.load_checkpoint(self.logdir, self)

    @property
    def epsilon_old(self):
        # hardcode this for now
        t1 = 1e5
        t2 = 1e6
        y1 = 1.0
        y2 = 0.1
        t = (self.num_steps - t1) / (t2 - t1)
        return y1 + (y2-y1) * np.clip(t, 0, 1)

    def update_target(self):
        self.target_model.load_state_dict(self.training_model.state_dict())

    def run_test_envs(self):
        # Just run one episode of each test environment.
        # Assumes that the environments themselves handle logging.
        for env in self.testing_envs:
            state = env.reset()
            done = False
            while not done:
                state = torch.tensor([state], device=self.compute_device, dtype=torch.float32)
                qvals = self.training_model(state).detach().cpu().numpy().ravel()
                state, reward, done, info = env.step(np.argmax(qvals))

    def collect_data(self):
        states = [
            e.last_state if hasattr(e, 'last_state') else e.reset()
            for e in self.training_envs
        ]
        tensor_states = torch.tensor(states, device=self.compute_device, dtype=torch.float32)
        qvals = self.training_model(tensor_states).detach().cpu().numpy()

        num_states, num_actions = qvals.shape
        actions = np.argmax(qvals, axis=-1)
        random_actions = np.random.randint(num_actions, size=num_states)
        use_random = np.random.random(num_states) < self.epsilon
        actions = np.choose(use_random, [actions, random_actions])

        for env, state, action in zip(self.training_envs, states, actions):
            next_state, reward, done, info = env.step(action)
            if done:
                next_state = env.reset()
                self.num_episodes += 1
            env.last_state = next_state
            self.replay_buffer.push(state, action, reward, next_state, done)

        self.num_steps += len(states)

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
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = torch.mean((q_value - expected_q_value)**2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        writer = self.summary_writer
        n = self.num_steps
        if report and self.summary_writer is not None:
            writer.add_scalar("loss", loss.item(), n)
            writer.add_scalar("epsilon", self.epsilon, n)
            writer.add_scalar("qvals/model_mean", q_values.mean().item(), n)
            writer.add_scalar("qvals/model_max", q_values.max(1)[0].mean().item(), n)
            writer.add_scalar("qvals/target_mean", next_q_values.mean().item(), n)
            writer.add_scalar("qvals/target_max", next_q_value.mean().item(), n)
            writer.flush()

    def train(self, steps):
        needs_report = True

        for _ in range(int(steps / len(self.training_envs))):
            num_steps = self.num_steps
            next_opt = round_up(num_steps, self.optimize_freq)
            next_update = round_up(num_steps, self.target_update_freq)
            next_checkpoint = round_up(num_steps, self.checkpoint_freq)
            next_report = round_up(num_steps, self.report_freq)
            next_test = round_up(num_steps, self.test_freq)

            self.collect_data()

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

            if num_steps >= next_checkpoint:
                checkpointing.save_checkpoint(self.logdir, self, [
                    'training_model', 'target_model', 'optimizer',
                ])

            if num_steps >= next_test:
                self.run_test_envs()
