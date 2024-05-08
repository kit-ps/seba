from .abstract import AbstractFaceAnonymization
from ...lib.utils import exec_ext_cmd

import os
import shutil


class FawkesAnonymization(AbstractFaceAnonymization):
    """Apply a fawkes anonymization to the face in an image
    Documentation: https://sandlab.cs.uchicago.edu/fawkes/
    Source Code: https://github.com/Shawn-Shan/fawkes

    Requires fawkes to be available and executable on the local system.
    For an automated setup in an venv, see scripts/install_fawkes.sh

    Required pips:
        none

    Parameters:
        - (string) mode: cloak generation mode, one of min, low, mid, high

    Options: (Parameters["opt"]; do not influece output)
        - (string) bin: location of the fawkes executable (required)
        - (int) batch_size: number of images to run together
        - (int) gpu: id of gpu to use
        - (int) max_img_batch: maximum number of images to pass to fawkes simultanously (reduce this if fawkes is running out-of-memory!)
    """

    name = "fawkes"

    def validate_config(self):
        if "opt" not in self.config or "bin" not in self.config["opt"]:
            raise AttributeError("FawkesAnonymization requires location of fawkes executable")

        if "mode" not in self.config:
            self.config["mode"] = "mid"

        if "batch_size" not in self.config["opt"]:
            self.config["opt"]["batch_size"] = "1"

        if "gpu" not in self.config["opt"]:
            self.config["opt"]["gpu"] = None

        if "max_img_batch" not in self.config["opt"]:
            self.config["opt"]["max_img_batch"] = 10000

        if "no_align" not in self.config:
            self.config["no_align"] = False

    def run_fawkes(self, folder, ext="jpg"):
        cmd = [
            self.config["opt"]["bin"],
            "-d",
            folder,
            "-m",
            self.config["mode"],
            "--batch-size",
            str(self.config["opt"]["batch_size"]),
            "--format",
            ext,
            "--separate_target",
        ]
        if self.config["no_align"]:
            cmd.append("--no-align")
        if self.config["opt"]["gpu"] is not None:
            cmd += ["-g", str(self.config["opt"]["gpu"])]
        exec_ext_cmd(cmd)

    def anonymize_all(self):
        if len(self.dataset.datapoints) <= self.config["opt"]["max_img_batch"]:
            self.run_fawkes(self.dataset.folder, ext=list(self.dataset.datapoints.values())[0].ext)
            for point in self.dataset.datapoints.values():
                fawkes_name = self.ffilename(point.get_path(), point.ext)
                os.replace(fawkes_name, point.get_path())
        else:
            # copy max_img_batch many imgs to a 'batch'-subfolder, run fawkes there,
            # move the cloaked imgs back, delete the subfolder.
            # repeat this until all imgs were cloaked.
            i = 0
            batch_folder = os.path.join(self.dataset.folder, "batch")

            while i * self.config["opt"]["max_img_batch"] < len(self.dataset.datapoints):
                batch = list(self.dataset.datapoints.values())[
                    i * self.config["opt"]["max_img_batch"] : (i + 1) * self.config["opt"]["max_img_batch"]
                ]
                os.mkdir(batch_folder)
                for point in batch:
                    path = point.get_path().split("/")
                    path.insert(-1, "batch")
                    os.symlink(point.get_path(), "/".join(path))
                self.run_fawkes(batch_folder, ext=batch[0].ext)
                for point in batch:
                    path = point.get_path().split("/")
                    path.insert(-1, "batch")
                    path = self.ffilename("/".join(path), point.ext)
                    os.replace(path, point.get_path())
                shutil.rmtree(batch_folder)
                i += 1

    def ffilename(self, name, ext="jpg"):
        return name.replace("." + ext, "_cloaked." + ext).replace(".jpg", ".jpeg")
