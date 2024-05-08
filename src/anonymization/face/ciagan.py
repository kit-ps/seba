from .abstract import AbstractFaceAnonymization
from ...lib.utils import exec_ext_cmd

import os
import shutil
import cv2
import numpy as np


class CiaganAnonymization(AbstractFaceAnonymization):
    """Apply a CIAGAN anonymization to the face in an image
    Documentation: https://github.com/dvl-tum/ciagan
    Paper: https://arxiv.org/pdf/2005.09544v2.pdf

    Requires CIAGAN to be available and executable on the local system.
    For an automated setup in an venv, see scripts/install_ciagan.sh

    Required pips:
        none

    Options: (Parameters["opt"]; do not influece output)
        - (string) bin: location of the CIAGAN executable (required)
        - (string) dlib: location of the dlib shape predictor (folder only) (required)
        - (string) model: location of pretrained model (without ext) (required)
        - (int) batch_size: number of images to process in one batch (optional, default 15000)
    """

    name = "ciagan"

    def validate_config(self):
        if "opt" not in self.config or "bin" not in self.config["opt"]:
            raise AttributeError("CiaganAnonymization requires location of ciagan executable")

        if "opt" not in self.config or "dlib" not in self.config["opt"]:
            raise AttributeError("CiaganAnonymization requires location of dlib shape predictor")

        if "opt" not in self.config or "model" not in self.config["opt"]:
            raise AttributeError("CiaganAnonymization requires location of model")

        if "opt" not in self.config or "batch_size" not in self.config["opt"]:
            self.config["opt"]["batch_size"] = 15000

    def anonymize_all(self):
        i = 0
        while i * self.config["opt"]["batch_size"] < len(self.dataset.datapoints):
            batch = list(self.dataset.datapoints.values())[
                i * self.config["opt"]["batch_size"] : (i + 1) * self.config["opt"]["batch_size"]
            ]
            self.anonymize_batch(batch)
            i += 1

    def anonymize_batch(self, batch):
        batch_folder = os.path.join(self.dataset.folder, "batch")
        os.mkdir(batch_folder)
        os.mkdir(os.path.join(batch_folder, "orig"))
        os.mkdir(os.path.join(batch_folder, "orig", "0"))
        os.mkdir(os.path.join(batch_folder, "processed"))
        os.mkdir(os.path.join(batch_folder, "anon"))

        index = 0
        for point in batch:
            newpath = os.path.join(batch_folder, "orig", "0", str(index).zfill(6) + ".jpg")
            point.tmp_index = index
            img = cv2.imread(point.get_path())
            cv2.imwrite(newpath, img)
            index += 1

        cmd = [
            self.config["opt"]["bin"],
            "process_data.py",
            "--input",
            batch_folder + "/orig",
            "--output",
            batch_folder + "/processed/",
            "--dlib",
            self.config["opt"]["dlib"],
        ]
        exec_ext_cmd(cmd)

        cmd = [
            self.config["opt"]["bin"],
            "test.py",
            "--model",
            self.config["opt"]["model"],
            "--data",
            batch_folder + "/processed/",
            "--out",
            batch_folder + "/anon/",
        ]
        exec_ext_cmd(cmd)

        for point in batch:
            path = os.path.join(batch_folder, "anon", str(point.tmp_index).zfill(6) + ".jpg")
            if os.path.exists(path):
                img = cv2.imread(path)
                img = self.remove_duplicated_pixels(img)
                img = cv2.resize(img, (224, 224), cv2.INTER_CUBIC)
                cv2.imwrite(point.get_path(), img)

        shutil.rmtree(batch_folder)

    def remove_duplicated_pixels(self, img):
        top = left = 0
        bottom, right = img.shape[:2]

        for i in range(img.shape[0] // 2):
            if np.absolute((img[i + 1][img.shape[1] // 2] - img[i][img.shape[1] // 2]).astype(np.int8)).sum() > 9:
                top = i
                break
        for i in range(img.shape[0] - 2, img.shape[0] // 2 + 1, -1):
            if np.absolute((img[i + 1][img.shape[1] // 2] - img[i][img.shape[1] // 2]).astype(np.int8)).sum() > 9:
                bottom = i
                break
        for i in range(img.shape[1] // 2):
            if np.absolute((img[img.shape[0] // 2][i + 1] - img[img.shape[0] // 2][i]).astype(np.int8)).sum() > 9:
                left = i
                break
        for i in range(img.shape[1] - 2, img.shape[1] // 2 + 1, -1):
            if np.absolute((img[img.shape[0] // 2][i + 1] - img[img.shape[0] // 2][i]).astype(np.int8)).sum() > 9:
                right = i
                break
        return img[top:bottom, left:right]
