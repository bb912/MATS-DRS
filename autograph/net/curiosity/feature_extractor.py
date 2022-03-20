import torch
from torch import nn as nn
from torch.nn import Module

from autograph.net.curiosity.utils import Flatten


class FeatureExtractor(Module):
    def __init__(self, in_channels, state_size, linear_input_size, feature_size, init_stride=2):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=5, stride=init_stride),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ELU(),
            Flatten()
        )

        out_size = self.main(torch.zeros(1, in_channels, *state_size)).shape[1] + linear_input_size

        self.last_linear = nn.Sequential(
            nn.Linear(out_size, feature_size * 2),
            nn.ReLU(),
            nn.Linear(feature_size * 2, feature_size),
            nn.Sigmoid()
        )

    def forward(self, state, lin_input=None):
        after_main = self.main(state)
        if lin_input is not None:
            after_main = torch.cat((after_main, lin_input), dim=-1)
        return self.last_linear(after_main)
