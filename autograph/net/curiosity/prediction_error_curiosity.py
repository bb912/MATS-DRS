from abc import ABC

from autograph.net.curiosity.curiosity_module import CuriosityModule


class PredictionErrorCuriosity(CuriosityModule, ABC):
    def __init__(self, predictor_net, target_net):
        super().__init__()
        self.predictor = predictor_net
        self.target = target_net

    def forward(self, next_state):
        return self.predictor(next_state), self.target(next_state)

    def get_intrinsic_reward(self, state, action, next_state):
        return self.get_training_loss(state, action, next_state).detach()
