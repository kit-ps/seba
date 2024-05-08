from .torch import TorchDeanonymization

import torch


class ApplymodelDeanonymization(TorchDeanonymization):
    """De-Anonymize faces using a pre-defined torch model

    Required pips:
        - torch
        - opencv-python
        - numpy

    Parameters:
        none
    """

    name = "applymodel"

    def validate_config(self):
        if "model" not in self.config:
            raise AttributeError("Model path required")

    def train(self, clear_set, anon_set):
        self.model = torch.load(self.config["model"])
