from net.curiosity.curiosity_module import CuriosityModule
from net.curiosity.feature_extractor import FeatureExtractor
from net.curiosity.utils import l2_norm_squared


class SAND(CuriosityModule):

    def __init__(self, in_channels, state_size, action_size, feature_size):
        super().__init__()
        self.predictor = FeatureExtractor(in_channels, state_size, action_size, feature_size)
        self.target = FeatureExtractor(in_channels, state_size, action_size, feature_size)

    def forward(self, state, action):
        return self.predictor(state, action), self.target(state, action)

    def get_intrinsic_reward(self, state, action, next_state):
        return self.get_training_loss(state, action, next_state).detach()

    def get_training_loss(self, state, action, next_state):
        predict, target = self(state, action)
        return l2_norm_squared(predict, target.detach())
