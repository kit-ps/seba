from .torch import TorchDeanonymization

import torch
import torch.nn as nn

from torchvision.transforms import Resize, InterpolationMode


class ConvAutoencoderModule(nn.Module):
    def __init__(self, channels=1, features=32, input_size=64):
        super(ConvAutoencoderModule, self).__init__()
        self.resize = False

        # padding & output padding in the convtranspose layers in the decoder may need to be adapted for other input dims
        apad = {
            # input: [first: pad, outputpad; second: pad, outputpad]
            16: [[1, 0], [0, 1]],
            24: [[1, 0], [0, 1]],
            32: [[1, 0], [0, 1]],
            64: [[1, 0], [0, 1]],
            224: [[1, 0], [0, 1]],
        }
        pad = apad[input_size]
        if input_size in [16, 24, 32]:
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
            nn.ConvTranspose2d(
                in_channels=features, out_channels=features, kernel_size=(3, 3), stride=2, padding=pad[0][0], output_padding=pad[0][1]
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=features, out_channels=features, kernel_size=(3, 3), stride=2, padding=pad[1][0], output_padding=pad[1][1]
            ),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=(3, 3), padding="same"),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.resize:
            x = self.resizer(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def code(self, x):
        return self.encoder(x)


class ConvaeDeanonymization(TorchDeanonymization):
    """De-Anonymize faces using a convolutional auto encoder

    Based on: https://keras.io/examples/vision/autoencoder/

    Required pips:
        - torch
        - opencv-python
        - numpy

    Parameters:
        none
    """

    name = "convae"

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
        self.model = ConvAutoencoderModule(channels=self.input_dim[0], features=self.config["features"], input_size=self.input_dim[1])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"]
        )
