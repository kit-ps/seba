from .abstract import AbstractFaceDeanonymization
from ...lib.utils import exec_ext_cmd

import os
import shutil


class StripformerDeanonymization(AbstractFaceDeanonymization):
    """De-Anonymize faces using Stripformer deblurring

    Requires installed implementation of paper in bin/stripformer.
    Install using installation script in scripts/installer/stripformer.sh
    Also, download pretrained CelebA DIC model and provide the path in model param.

    Code: https://github.com/pp00704831/Stripformer

    Paper:
        Fu-Jen Tsai and Yan-Tsung Peng and Yen-Yu Lin and Chung-Chi Tsai and Chia-Wen Lin
        Stripformer: Strip Transformer for Fast Image Deblurring

    Required pips:
        none

    Parameters:
        - opt['bin'] (string): location of Stripformer executable
    """

    name = "stripformer"

    def validate_config(self):
        if "opt" not in self.config or "bin" not in self.config["opt"]:
            raise AttributeError("Stripformer Deanonymization requires location of executable")

    def deanonymize_all(self):
        os.mkdir(os.path.join(self.dataset.folder, "batch"))
        os.mkdir(os.path.join(self.dataset.folder, "batch", "0"))

        for point in self.dataset.datapoints.values():
            path = self.new_path(point.get_path())
            shutil.copyfile(point.get_path(), path)

        cmd = [
            "env/bin/python3",
            "predict_GoPro_test_results.py",
            "--weights_path",
            "Weights/Stripformer_gopro.pth",
            "--blur_path",
            os.path.join(self.dataset.folder, "batch"),
            "--out_path",
            os.path.join(self.dataset.folder, "batch"),
        ]
        exec_ext_cmd(cmd, cwd=self.config["opt"]["bin"])

        for point in self.dataset.datapoints.values():
            path = self.new_path(point.get_path())
            os.replace(path, point.get_path())

        shutil.rmtree(os.path.join(self.dataset.folder, "batch"))

    def new_path(self, oldpath):
        path = oldpath.split("/")
        path.insert(-1, "batch/0")
        path = "/".join(path)
        return path
