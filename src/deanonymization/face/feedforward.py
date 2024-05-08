from .torch import TorchDeanonymization

import torch
import torch.nn as nn
import math


class FeedforwardModule(nn.Module):
    def __init__(self, input_dim=2):
        super(FeedforwardModule, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x):
        x = self.ff(x)
        return x


class FeedforwardDeanonymization(TorchDeanonymization):
    """De-Anonymize faces using a ML feed-forwarding

    Required pips:
        - torch
        - opencv-python
        - numpy

    Parameters:
        none
    """

    name = "feedforward"

    def validate_config(self):
        super().validate_config()

    def create_model(self):
        self.input_dim = [math.prod(self.input_shape)]
        self.model = FeedforwardModule(input_dim=self.input_dim[0])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"]
        )
