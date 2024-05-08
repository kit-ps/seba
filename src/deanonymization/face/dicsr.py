from .abstract import AbstractFaceDeanonymization
from ...lib.utils import exec_ext_cmd

import json
import os
import shutil
import cv2


class DicsrDeanonymization(AbstractFaceDeanonymization):
    """De-Anonymize faces using Deep Face Super-Resolution with Iterative Collaboration

    Requires installed implementation of paper in bin/dicsr.
    Install using installation script in scripts/install_dicsr.sh
    Also, download pretrained CelebA DIC model and provide the path in model param.

    Code: https://github.com/Maclory/Deep-Iterative-Collaboration

    Paper:
        Ma, Cheng and Jiang, Zhenyu and Rao, Yongming and Lu, Jiwen and Zhou, Jie.
        Deep Face Super-Resolution with Iterative Collaboration between Attentive Recovery and Landmark Estimation.

    Required pips:
        - opencv2

    Parameters:
        - model (string): The SR model to use: one of DIC, DICGAN (optional, default: DIC)
        - final_res (int): the final resolution of the image (=width=height) (optional, default 224)
        - opt['bin'] (string): location of DIC SR executable
        - opt['model_path'] (string): location of DIC SR CelebA DIC model pth
    """

    name = "dicsr"

    def validate_config(self):
        if "model" not in self.config:
            self.config["model"] = "DIC"
        else:
            if self.config["model"] not in ["DIC", "DICGAN"]:
                raise AttributeError("DIC SR Deanonymization model must be one of DIC, DICGAN")

        if "final_res" not in self.config:
            self.config["final_res"] = 224
        else:
            self.config["final_res"] = int(self.config["final_res"])

        if "opt" not in self.config or "bin" not in self.config["opt"]:
            raise AttributeError("DIC SR Deanonymization requires location of executable")

        if "opt" not in self.config or "model_path" not in self.config["opt"]:
            raise AttributeError("DIC SR Deanonymization requires location of pretrained model")

    def deanonymize_all(self):
        self.write_config()
        cmd = [self.config["opt"]["bin"], "-opt", os.path.join(self.dataset.folder, "config.json")]
        exec_ext_cmd(cmd)

        for point in self.dataset.datapoints.values():
            path = point.get_path().split("/")
            path.insert(-1, "results/DIC_in3f48_x8_celeba/celeba")
            path = "/".join(path)
            os.replace(path, point.get_path())

        shutil.rmtree(os.path.join(self.dataset.folder, "results"))
        os.remove(os.path.join(self.dataset.folder, "config.json"))

    def write_config(self):
        filename = os.path.join(self.dataset.folder, "config.json")
        folder = self.dataset.folder

        img = cv2.imread(list(self.dataset.datapoints.values())[0].get_path())
        width = img.shape[0]

        base = {
            "name": "celeba",
            "mode": "sr_align",
            "degradation": "BI",
            "use_gpu": False,
            "use_tb_logger": False,
            "scale": 8,
            "is_train": False,
            "rgb_range": 1,
            "save_image": True,
            "datasets": {"test_celeba": {"mode": "LR", "name": "celeba", "dataroot_LR": "BASE", "data_type": "img", "LR_size": 32}},
            "networks": {
                "which_model": "DIC",
                "num_features": 48,
                "in_channels": 3,
                "out_channels": 3,
                "num_steps": 4,
                "num_groups": 6,
                "detach_attention": False,
                "hg_num_feature": 256,
                "hg_num_keypoints": 68,
                "num_fusion_block": 7,
            },
            "solver": {"pretrained_path": "MODEL_PATH"},
            "path": {"root": "BASE"},
        }

        base["datasets"]["test_celeba"]["dataroot_LR"] = folder
        base["datasets"]["test_celeba"]["LR_size"] = width
        if self.config["model"] == "DICGAN":
            base["mode"] = "sr_align_gan"
        base["solver"]["pretrained_path"] = self.config["opt"]["model_path"]
        base["path"]["root"] = folder

        with open(filename, "w") as f:
            f.write(json.dumps(base))
