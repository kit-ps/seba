from .torch import TorchDeanonymization

import torch
import torch.nn as nn

from torchvision.transforms import Resize, InterpolationMode


class ConvlinAutoencoderModule(nn.Module):
    def __init__(self, channels=1, features=4, input_size=224):
        super(ConvlinAutoencoderModule, self).__init__()
        self.features = features
        self.resize = False

        if not input_size == 224:
            self.resize = True
            self.resizer = Resize((224, 224), interpolation=InterpolationMode.NEAREST)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=features, out_channels=features, kernel_size=(3, 3), stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=features, out_channels=features, kernel_size=(3, 3), stride=2, padding=0, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=(3, 3), padding="same"),
            nn.Sigmoid(),
        )
        self.lin = nn.Sequential(
            nn.Linear(features * 56 * 56, features * 56 * 56),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        if self.resize:
            x = self.resizer(x)
        x = self.encoder(x)
        x = self.lin(x.view(-1, self.features * 56 * 56))
        x = self.decoder(x.view(-1, self.features, 56, 56))
        return x

    def code(self, x):
        return self.encoder(x)


class ConvlinaeDeanonymization(TorchDeanonymization):
    """De-Anonymize faces using a convolutional auto encoder with one linear layer in the middle

    Required pips:
        - torch
        - opencv-python
        - numpy

    Parameters:
        none
    """

    name = "convlinae"

    def validate_config(self):
        super().validate_config()

        if "features" not in self.config:
            self.config["features"] = 9
        else:
            self.config["features"] = int(self.config["features"])

    def create_model(self):
        self.input_dim = list(self.input_shape)
        if len(self.input_dim) == 2:
            self.input_dim.insert(0, 1)
        self.model = ConvlinAutoencoderModule(channels=self.input_dim[0], features=self.config["features"], input_size=self.input_dim[1])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"]
        )
