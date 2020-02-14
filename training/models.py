import numpy as np

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


def signed_sqrt(x):
    s = torch.sign(x)
    return s * torch.sqrt(torch.abs(x))


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, factorized=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.factorized = factorized

        init_scale = in_features**-0.5
        self.weight_mu = nn.Parameter(
            2 * init_scale * (torch.rand(out_features, in_features)-0.5))
        self.weight_sigma = nn.Parameter(
            2 * init_scale * (torch.rand(out_features, in_features)-0.5))
        if self.use_bias:
            self.bias_mu = nn.Parameter(
                2 * init_scale * (torch.rand(out_features)-0.5))
            self.bias_sigma = nn.Parameter(
                2 * init_scale * (torch.rand(out_features)-0.5))

    def forward(self, x):
        b = None
        device = self.weight_mu.device
        if self.factorized:
            eps1 = signed_sqrt(torch.randn(self.in_features, device=device))
            eps2 = signed_sqrt(torch.randn(self.out_features, device=device))
            w = self.weight_mu + self.weight_sigma * eps1 * eps2[:,np.newaxis]

            if self.use_bias:
                # As with the original paper, use the signed sqrt even though
                # we're not taking a product of noise params.
                eps3 = signed_sqrt(torch.randn(self.out_features, device=device))
                b = self.bias_mu + self.bias_sigma * eps3

        else:
            eps1 = torch.randn(self.out_features, self.in_features, device=device)
            w = self.weight_mu + self.weight_sigma * eps1

            if self.use_bias:
                eps3 = torch.randn(self.out_features, device=device)
                b = self.bias_mu + self.bias_sigma * eps3

        return F.linear(x, w, b)


class SafeLifeQNetwork(nn.Module):
    """
    Module for calculating Q functions.
    """
    def __init__(self, input_shape, use_noisy_layers=True):
        super().__init__()

        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        num_features = np.product(cnn_out_shape)
        num_actions = 9

        Linear = NoisyLinear if use_noisy_layers else nn.Linear

        self.advantages = nn.Sequential(
            Linear(num_features, 256),
            nn.ReLU(),
            Linear(256, num_actions)
        )

        self.value_func = nn.Sequential(
            Linear(num_features, 256),
            nn.ReLU(),
            Linear(256, 1)
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
    def __init__(self, input_shape):
        super().__init__()

        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        num_features = np.product(cnn_out_shape)
        num_actions = 9

        self.dense = nn.Sequential([
            nn.Linear(num_features, 512),
            nn.ReLU(),
        ])
        self.logits = nn.Linear(512, num_actions)
        self.value_func = nn.Linear(512, 1)

    def forward(self, obs):
        # Switch observation to (c, w, h) instead of (h, w, c)
        obs = obs.transpose(-1, -3)
        x = self.cnn(obs).flatten(start_dim=1)
        x = self.dense(x)
        value = self.value_func(x)[...,0]
        advantages = F.softmax(self.logits(x), dim=-1)
        return value, advantages
