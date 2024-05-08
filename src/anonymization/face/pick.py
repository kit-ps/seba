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

        if "hardlink" not in self.config:
            self.config["hardlink"] = False

    def anonymize(self, image):
        filename = image.get_path().split("/")[-1]
        path = os.path.join(os.getcwd(), "data", self.config["dataset"], filename)
        self.replace_file(image.get_path(), path)

        self.replace_file(image.get_path().replace(image.ext, "yaml"), path.replace(image.ext, "yaml"), ignore=True)

        if image.idname not in self.ids:
            self.ids.append(image.idname)
            old = os.path.join(image.setpath, image.idname + ".yaml")
            new = os.path.join(os.getcwd(), "data", self.config["dataset"], image.idname + ".yaml")
            self.replace_file(old, new, ignore=True)

    def replace_file(self, old, new, ignore=False):
        try:
            os.remove(old)
            if self.config["hardlink"]:
                shutil.copy(new, old)
            else:
                os.symlink(new, old)
        except Exception:
            if ignore:
                return
            else:
                raise ValueError("Failed to replace file " + old + " with " + new)
