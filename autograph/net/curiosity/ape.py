from net.curiosity.prediction_error_curiosity import PredictionErrorCuriosity
from net.curiosity.utils import l2_norm_squared


class APE(PredictionErrorCuriosity):
    def get_training_loss(self, state, action, next_state):
        (pred_policy, pred_value), (act_policy, act_value) = self(next_state)
        return l2_norm_squared(pred_policy, act_policy.detach())
