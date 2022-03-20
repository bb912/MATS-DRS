from autograph.net.curiosity.feature_extractor import FeatureExtractor
from autograph.net.curiosity.prediction_error_curiosity import PredictionErrorCuriosity
from autograph.net.curiosity.utils import l2_norm_squared


class RND(PredictionErrorCuriosity):

    def __init__(self, in_channels, state_size, feature_size, init_stride=2):
        super().__init__(FeatureExtractor(in_channels, state_size, 0, feature_size, init_stride),
                         FeatureExtractor(in_channels, state_size, 0, feature_size, init_stride))

        for p in self.target.parameters():
            p.requires_grad = False

    def get_training_loss(self, state, action, next_state):
        predict, target = self(next_state)
        return l2_norm_squared(predict, target.detach())
