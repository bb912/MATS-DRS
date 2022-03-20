import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

from net.curiosity.curiosity_module import CuriosityModule
from net.curiosity.feature_extractor import FeatureExtractor
from net.curiosity.utils import l2_norm_squared

BETA = 0.2
LAMBDA = 0.1


class Forward(Module):
    def __init__(self, feature_size, num_actions):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(feature_size + num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_size)
        )

    def forward(self, features, action):
        return self.main(torch.cat((features, action), dim=-1))


class Inverse(Module):
    def __init__(self, feature_size, num_actions):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(feature_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, features, next_features):
        return self.main(torch.cat((features, next_features), dim=-1))


class ICM(CuriosityModule):
    def __init__(self, in_channels, state_size, feature_size, num_actions):
        super().__init__()

        self.extractor = FeatureExtractor(in_channels, state_size, 0, feature_size)
        self.forward_model = Forward(feature_size, num_actions)
        self.inverse_model = Inverse(feature_size, num_actions)

    def forward(self, state, action, next_state):
        features = self.extractor(state)
        next_features = self.extractor(state)
        action_prediction = self.inverse_model(features, next_features)
        next_features_prediction = self.forward_model(features, action)
        return features, next_features, action_prediction, next_features_prediction

    def get_intrinsic_reward(self, state, action, next_state):
        _, next_features, _, next_prediction = self(state, action, next_state)
        # Magnitude of vector difference
        pred_loss = l2_norm_squared(next_features, next_prediction) * .5
        return pred_loss.detach()

    def get_training_loss(self, state, action, next_state):
        features, next_features, action_prediction, next_features_prediction = self(state, action, next_state)

        inv_loss = (-F.log_softmax(action_prediction, dim=-1) * action).sum()
        forward_loss = l2_norm_squared(next_features, next_features_prediction) * .5
        return (1 - BETA) * inv_loss + BETA * forward_loss
