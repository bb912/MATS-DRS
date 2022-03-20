import torch.nn as nn


class BasicNet(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size):
        super().__init__()

        self.policy_head = nn.Sequential(
            nn.Linear(input_size, intermediate_size),
            nn.LeakyReLU(),
            nn.LayerNorm(intermediate_size),
            nn.Linear(intermediate_size, intermediate_size),
            nn.LeakyReLU(),
            nn.LayerNorm(intermediate_size),
            nn.Linear(intermediate_size, output_size)
        )

        self.value_head = nn.Sequential(
            nn.Linear(input_size, intermediate_size),
            nn.LeakyReLU(),
            nn.LayerNorm(intermediate_size),
            nn.Linear(intermediate_size, intermediate_size),
            nn.LeakyReLU(),
            nn.LayerNorm(intermediate_size),
            nn.Linear(intermediate_size, 1)
        )

    def forward(self, input):
        return self.policy_head(input), self.value_head(input)
