import numpy as np

from training.utils import recursive_shape, check_shape

import torch

from torch import nn
from torch.nn import functional as F


def safelife_cnn(input_shape):
    """
    Defines a CNN with good default values for safelife.

    This works best for inputs of size 25x25.

    Parameters
    ----------
    input_shape : tuple of ints
        Height, width, and number of channels for the board.

    Returns
    -------
    cnn : torch.nn.Sequential
    output_shape : tuple of ints
        Channels, width, and height.

    Returns both the CNN module and the final output shape.
    """
    h, w, c = input_shape
    cnn = nn.Sequential(
        nn.Conv2d(c, 32, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU()
    )
    h = (h-4+1)//2
    h = (h-2+1)//2
    h = (h-2)
    w = (w-4+1)//2
    w = (w-2+1)//2
    w = (w-2)
    return cnn, (64, w, h)


class SafeLifeQNetwork(nn.Module):
    """
    Module for calculating Q functions.
    """
    def __init__(self, input_shape):
        super().__init__()

        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        num_features = np.product(cnn_out_shape)
        num_actions = 9

        self.advantages = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        self.value_func = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        # Switch observation to (c, w, h) instead of (h, w, c)
        obs = obs.transpose(-1, -3)
        x = self.cnn(obs).flatten(start_dim=1)
        advantages = self.advantages(x)
        value = self.value_func(x)
        qval = value + advantages - advantages.mean()
        return qval


class SafeLifePolicyNetwork(nn.Module):
    def __init__(self, input_shape, dense_depth=1):
        super().__init__()

        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        num_features = np.product(cnn_out_shape)
        num_actions = 9

        dense = [nn.Sequential(nn.Linear(num_features, 512), nn.ReLU())]
        for n in range(dense_depth - 1):
            dense.append(nn.Sequential(nn.Linear(512, 512), nn.ReLU()))
        self.dense = nn.Sequential(*dense)

        self.logits = nn.Linear(512, num_actions)
        self.value_func = nn.Linear(512, 1)

    def forward(self, obs):
        # Switch observation to (c, w, h) instead of (h, w, c)
        obs = obs.transpose(-1, -3)
        x = self.cnn(obs).flatten(start_dim=1)
        for layer in self.dense:
            x = layer(x)
        value = self.value_func(x)[...,0]
        policy = F.softmax(self.logits(x), dim=-1)
        return value, policy

class SafeLifeLSTMPolicyNetwork(nn.Module):
    def __init__(self, input_shape, dense_depth=1, lstm_shape=(3,1), dropout=0):
        super().__init__()
        lstm_channels, lstm_depth = lstm_shape

        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        h, w, c = cnn_out_shape
        num_features = np.product(cnn_out_shape)
        num_actions = 9

        dense = [nn.Sequential(nn.Linear(num_features + h * w * lstm_channels, 512), nn.ReLU())]
        for n in range(dense_depth - 1):
            dense.append(nn.Sequential(nn.Linear(512, 512), nn.ReLU()))
        self.dense = nn.Sequential(*dense)

        self.memory = nn.LSTM(
            input_size=num_features, hidden_size=h * w * lstm_channels, num_layers=lstm_depth, 
            dropout=dropout,
        )

        self.logits = nn.Linear(512, num_actions)
        self.value_func = nn.Linear(512, 1)

    def forward(self, obs, state):
        check_shape(state, ("*", 2, 1,  576), "state passed to LSTM PPO")

        # Switch observation to (c, w, h) instead of (h, w, c)
        obs = obs.transpose(-1, -3)

        x = self.cnn(obs).flatten(start_dim=1)
        # print("CNN shape", x.shape)
        y = x[np.newaxis, ...]
        # print("Memory shape", y.shape)
        # import inspect
        # curframe = inspect.currentframe()
        # calframe = inspect.getouterframes(curframe, 2)
        # caller = calframe[1][3]
        # print("Called from", caller)
        print("Applying memory with data", obs.shape, "and state:", recursive_shape(state))
        state = state.permute(1, 2, 0, 3)  # move (hs, cs) to front
        y = self.memory(y, tuple(state))
        # print(recursive_shape(y))
        y, state = y
        state = torch.stack(state)  # (hs, cs) tuple -> tensor
        state = state.permute(2, 0, 1, 3)  # move batch to front
        print("return shape is", state.shape)
        y = y[0] # remove time
        # print("Merging", x.shape, y.shape)
        merged = torch.cat((x, y), dim=1)
        # print("Merged shape", merged.shape)
        x = self.dense[0](merged)

        rest = self.dense[1:]
        for layer in rest:
            x = layer(x)
        value = self.value_func(x)[...,0]
        policy = F.softmax(self.logits(x), dim=-1)
        return value, policy, state
