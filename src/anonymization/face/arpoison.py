from .abstract import AbstractFaceAnonymization

import torch
from torchvision.io import read_image
from PIL import Image
from bin.arps.autoregressive import ARProcessPerturb3Channel


class ArpoisonAnonymization(AbstractFaceAnonymization):
    """Adds autoregressive poisoning to images

    Requires installed autoregressive poisoning library installed in ./bin.
    Install in bin/ via:
        git clone https://github.com/psandovalsegura/autoregressive-poisoning.git arps

    Required pips:
        - opencv-python
        - torch

    Parameters:
        - (string) coefficients: path of coefficients file
        - (int) offset: int where to sart using coefficients
    """

    name = "arpoison"

    def validate_config(self):
        if "offset" not in self.config:
            self.config["offset"] = 0

        if "coefficients" not in self.config:
            raise AttributeError("ArPoisen requires path to coefficients file")

    def anonymize_all(self):
        coefficients = torch.load(self.config["coefficients"])
        i = self.config["offset"]

        for id in self.dataset.point_by_id():
            arp = ARProcessPerturb3Channel(b=coefficients[i])
            for imgid in id:
                img = read_image(self.dataset.datapoints[imgid].get_path()).div(255)
                delta, _ = arp.generate(size=(img.size()[1] + 4, img.size()[2] + 4), eps=1.0, crop=4, p=2)
                img = (img + delta).clamp(0, 1)
                img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                img = Image.fromarray(img)
                img.save(self.dataset.datapoints[imgid].get_path())

            i += 1
