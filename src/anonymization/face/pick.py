from .abstract import AbstractFaceAnonymization

import os
import shutil


class PickAnonymization(AbstractFaceAnonymization):
    """Apply an anonymization by picking the respective file form a specified dataset

    Required pips:
        none

    Parameters:
        - (string) dataset: set to pick anonymized files from
        - (bool) hardlink: whether to hardlink picked files to new set (optional, default: false)
    """

    name = "pick"

    def validate_config(self):
        self.ids = []
        if "dataset" not in self.config:
            raise AttributeError("PickAnonymization requires dataset to pick from")

        if "softlink" not in self.config:
            self.config["softlink"] = False

    def anonymize(self, image):
        image.replace(os.path.join(image.get_path().split("/")[:-2], self.config['dataset'], image.filename), soft=self.config['softlink'])
