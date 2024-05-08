from .abstract import AbstractFaceAnonymization
from ...lib.utils import exec_ext_cmd

import os
import shutil


class DeepprivacyAnonymization(AbstractFaceAnonymization):
    """Apply a DeepPrivacy anonymization to the face in an image
    Documentation: https://github.com/hukkelas/DeepPrivacy
    Paper: https://arxiv.org/abs/1909.04538

    Requires DeepPrivacy to be available and executable on the local system.
    For an automated setup in an venv, see scripts/install_deepprivacy.sh

    Required pips:
        none

    Parameters:
        - (string) model: the anonymization model to be used, one of
            {fdf128_rcnn512,fdf128_retinanet512,fdf128_retinanet256,fdf128_retinanet128,deep_privacy_V1}
            NOTE: fdf128_rcnn512 & deep_privacy_V1 use RCNN detector for face detection which might cause CUDA out-of-memory errors!

    Options: (Parameters["opt"]; do not influece output)
        - (string) bin: location of the DeepPrivacy executable (required)
    """

    name = "deepprivacy"

    def validate_config(self):
        if "opt" not in self.config or "bin" not in self.config["opt"]:
            raise AttributeError("DeepprivacyAnonymization requires location of deepprivacy executable")

        if "batch_size" not in self.config["opt"]:
            self.config["opt"]["max_img_batch"] = 2000

        if "model" not in self.config:
            self.config["model"] = "fdf128_retinanet256"  # Recommended in documentation.

    def anonymize_all(self):
        i = 0

        while i * self.config["opt"]["batch_size"] < len(self.dataset.datapoints):
            batch = list(self.dataset.datapoints.values())[
                i * self.config["opt"]["batch_size"] : (i + 1) * self.config["opt"]["batch_size"]
            ]
            self.run_batch(batch)
            i += 1

    def run_batch(self, batch):
        if len(batch) == 1:
            self.log.warn("Running DeepPrivacy with batch of size 1 does not work. Modify batch size!")
            return
        batch_folder = os.path.join(self.dataset.folder, "batch")
        os.mkdir(batch_folder)

        for point in batch:
            path = point.get_path().split("/")
            path.insert(-1, "batch")
            os.symlink(point.get_path(), "/".join(path))

        self.run_deepprivacy(batch_folder)
        shutil.rmtree(batch_folder)

    def run_deepprivacy(self, folder):
        cmd = [
            self.config["opt"]["bin"],
            "-m",
            self.config["model"],
            "-s",
            folder,
            "-t",
            folder,
        ]
        exec_ext_cmd(cmd)
