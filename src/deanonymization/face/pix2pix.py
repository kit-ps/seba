from .abstract import AbstractFaceDeanonymization
from ...lib.utils import exec_ext_cmd

import random
import os
import shutil
import cv2


class Pix2pixDeanonymization(AbstractFaceDeanonymization):
    """De-Anonymize faces by training a Pix2Pix model.

    Isola, Phillip, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros.
    "Image-to-image translation with conditional adversarial networks."
    In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1125-1134. 2017.

    Code: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/

    Requires installed pix2pix using scripts/installers/pix2pix.sh

    Required pips:
        none

    Parameters:
        - train_rate (float): split between training and validation data
        - epochs (int): number of epochs to train model
        - model (string): use other existing model.
        - opt['bin'] (string): location of Pix2Pix executable
        - opt['gpu_ids'] (int): IDs of GPUs to use. Use -1 for CPU-mode.
    """

    name = "pix2pix"

    def validate_config(self):
        if "opt" not in self.config or "bin" not in self.config["opt"]:
            raise AttributeError("Pix2Pix Deanonymization requires location of executable")

        if "opt" not in self.config or "gpu_ids" not in self.config["opt"]:
            raise AttributeError("Pix2Pix Deanonymization requires ids of gpus to use. Use -1 for CPU-mode.")

        if "model" not in self.config:
            if "train_rate" not in self.config:
                raise AttributeError("Pix2Pix Deanonymization requires train rate")

            if "epochs" not in self.config:
                raise AttributeError("Pix2Pix Deanonymization requires number of epochs")

    def train(self, clear_set, anon_set):
        # create folders
        self.clear = clear_set.name
        if "model" in self.config:
            self.clear = self.config["model"]
            return

        model_path = os.path.join(self.config["opt"]["bin"], "checkpoints", self.clear)
        if os.path.exists(model_path):
            return

        self.base = os.path.join(clear_set.folder, "pix2pix")
        os.mkdir(self.base)
        os.mkdir(os.path.join(self.base, "data"))
        self.create_datastructure(clear_set, anon_set)

        # create training data
        cmd = [
            "env/bin/python3",
            "datasets/combine_A_and_B.py",
            "--fold_A",
            os.path.join(self.base, "data", "A"),
            "--fold_B",
            os.path.join(self.base, "data", "B"),
            "--fold_AB",
            os.path.join(self.base, "data"),
        ]
        exec_ext_cmd(cmd, cwd=self.config["opt"]["bin"])

        # do training
        cmd = [
            "env/bin/python3",
            "train.py",
            "--dataroot",
            os.path.join(self.base, "data"),
            "--name",
            self.clear,
            "--model",
            "pix2pix",
            "--direction",
            "BtoA",
            "--gpu_ids",
            str(self.config["opt"]["gpu_ids"]),
            "--display_id",
            "0",
            "--no_html",
            "--n_epochs",
            str(self.config["epochs"] // 2),
            "--n_epochs_decay",
            str(self.config["epochs"] // 2),
        ]
        exec_ext_cmd(cmd, cwd=self.config["opt"]["bin"])

        shutil.rmtree(self.base)
        for file in os.listdir(model_path):
            if "latest" not in file:
                os.remove(os.path.join(model_path, file))

    def create_datastructure(self, clear_set, anon_set):
        for a in ["A", "B"]:
            os.mkdir(os.path.join(self.base, "data", a))
            for b in ["train", "val"]:
                os.mkdir(os.path.join(self.base, "data", a, b))

        # create softlinks
        random.seed(a="seed")
        keys = list(clear_set.datapoints.keys())
        random.shuffle(keys)
        index = int(self.config["train_rate"] * len(keys))

        target_shape = None
        ci = cv2.imread(clear_set.datapoints[keys[0]].get_path())
        ai = cv2.imread(anon_set.datapoints[keys[0]].get_path())
        if not ci.shape == ai.shape:
            target_shape = ci.shape

        self.copy_files(clear_set, anon_set, keys[:index], "train", target_shape)
        self.copy_files(clear_set, anon_set, keys[index:], "val", target_shape)

    def copy_files(self, clear, anon, keys, folder, target):
        for k in keys:
            os.symlink(clear.datapoints[k].get_path(), os.path.join(self.base, "data", "A", folder, clear.datapoints[k].get_filename()))
            if target:
                cv2.imwrite(
                    os.path.join(self.base, "data", "B", folder, anon.datapoints[k].get_filename()),
                    cv2.resize(cv2.imread(anon.datapoints[k].get_path()), target[:2], interpolation=cv2.INTER_NEAREST),
                )
            else:
                os.symlink(anon.datapoints[k].get_path(), os.path.join(self.base, "data", "B", folder, anon.datapoints[k].get_filename()))

    def deanonymize_all(self):
        cmd = [
            "env/bin/python3",
            "test.py",
            "--dataroot",
            self.dataset.folder,
            "--name",
            self.clear,
            "--model",
            "test",
            "--direction",
            "BtoA",
            "--dataset_mode",
            "single",
            "--gpu_ids",
            str(self.config["opt"]["gpu_ids"]),
            "--netG",
            "unet_256",
            "--norm",
            "batch",
            "--num_test",
            str(len(self.dataset.datapoints)),
        ]
        exec_ext_cmd(cmd, cwd=self.config["opt"]["bin"])

        results = os.path.join(self.config["opt"]["bin"], "results", self.clear, "test_latest", "images")
        for point in self.dataset.datapoints.values():
            path = os.path.join(results, point.get_filename().replace("." + point.ext, "_fake." + point.ext))
            os.replace(path, point.get_path())

        shutil.rmtree(os.path.join(self.config["opt"]["bin"], "results", self.clear))
