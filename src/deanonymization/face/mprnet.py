from .abstract import AbstractFaceDeanonymization
from ...lib.utils import exec_ext_cmd


class MprnetDeanonymization(AbstractFaceDeanonymization):
    """De-Anonymize faces using MPRnet

    Requires installed implementation of paper in bin/mprnet.
    Install using installation script in scripts/install_mprnet.sh

    Code: https://github.com/swz30/MPRNet

    Paper:
        Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
        Multi-Stage Progressive Image Restoration (CVPR 2021)

    Required pips:
        - opencv2

    Parameters:
        - (string) task: task argument for MPRnet (one of: Deblurring, Denoising)
        - opt['bin'] (string): location of MPRnet executable
    """

    name = "mprnet"

    def validate_config(self):
        if "task" not in self.config:
            raise AttributeError("MPRNet requires de-anonymization task, one of Deblurring or Denoising")

        if "opt" not in self.config or "bin" not in self.config["opt"]:
            raise AttributeError("MPRnet Deanonymization requires location of executable")

    def deanonymize_all(self):
        cmd = [
            "env/bin/python3",
            "demo.py",
            "--task",
            self.config["task"],
            "--input_dir",
            self.dataset.folder,
            "--result_dir",
            self.dataset.folder,
        ]
        exec_ext_cmd(cmd, cwd=self.config["opt"]["bin"])
