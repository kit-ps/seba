from ...lib.result import Result
from .abstract import AbstractFacePrivacy
from ...lib.inference import Classification
from ...lib.utils import exec_ext_cmd

import os
import os.path
import shutil
import uuid

import cv2
import numpy as np
import torch
from deepface.commons.distance import findCosineDistance


class ArcfaceClassification(Classification, AbstractFacePrivacy):
    """Train and use a ArcFace model.
    Implementation from https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch
    -> Install it using scripts/installer/arcface.sh

    Deng, Jiankang, Jia Guo, Niannan Xue, and Stefanos Zafeiriou.
    "Arcface: Additive angular margin loss for deep face privacy."
    In Proceedings of the IEEE/CVF conference on computer vision and pattern privacy, pp. 4690-4699. 2019.

    Required pips:
        torch, deepface, numpy

    Parameters:
        None
    """

    def validate_config(self):
        if "opt" not in self.config or "num_gpus" not in self.config["opt"]:
            raise AttributeError("ArcFace requires number of available gpus")

    def train(self, set):
        self.trainingset = set.name
        if os.path.exists("arcface_models/" + set.name + ".pt"):
            self.log.info("Skipping training from scratch - using cached model: " + set.name + ".pt")
            return

        self.id = str(uuid.uuid4())[:8]
        self.create_training_data(set)
        self.create_config(set)
        self.do_training(set.folder)
        self.training_cleanup()

    @torch.no_grad()
    def enroll(self, set):
        self.log.info("Enrolling folder " + set.name)
        self.net = self.load_model()
        self.reprs = {}

        for point in set.datapoints.values():
            img = self.load_img(point.get_path())
            self.reprs[point.idname + "." + point.pointname] = self.net(img).numpy()[0]

    @torch.no_grad()
    def classify_point(self, image):
        rs = Result(image.idname, image.pointname)
        img = self.load_img(image.get_path())
        repr = self.net(img).numpy()[0]

        for k, v in self.reprs.items():
            dst = findCosineDistance(v, repr)
            rs.add_recognized(k.split(".")[0], dist=dst)

        self.log.debug(str(rs))
        return rs

    @torch.no_grad()
    def get_encoding(self, image):
        img = self.load_img(image.get_path())
        return self.new(img).numpy()[0]

    def create_training_data(self, set):
        root_folder = os.path.join(set.folder, "root")
        os.mkdir(root_folder)

        for key in set.identities.keys():
            os.mkdir(os.path.join(root_folder, key))

        for point in set.datapoints.values():
            path = os.path.join(root_folder, point.idname, point.pointname + "." + point.ext)
            os.symlink(point.get_path(), path)

        cmd = ["python3", "-m", "mxnet.tools.im2rec", "--list", "--recursive", "train", str(root_folder)]
        exec_ext_cmd(cmd)
        cmd = [
            "python3",
            "-m",
            "mxnet.tools.im2rec",
            "--num-thread",
            "16",
            "--quality",
            "100",
            "--resize",
            "112",
            "train",
            str(root_folder),
        ]
        exec_ext_cmd(cmd)

        os.mkdir("bin/arcface/training-" + self.id)
        for f in ["train.lst", "train.rec", "train.idx"]:
            os.rename(f, os.path.join("bin/arcface/training-" + self.id, f))
        shutil.rmtree(root_folder)

    def create_config(self, set):
        s = """from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = "training-{}/"
config.num_classes = {}
config.num_image = {}
config.num_epoch = 40
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
""".format(
            self.id, len(set.identities), len(set.datapoints)
        )

        with open("bin/arcface/configs/conf-" + self.id + ".py", "w") as f:
            f.write(s)

    def do_training(self, setfolder):
        cmd = [
            "python3",
            "-m",
            "torch.distributed.launch",
            "--nproc_per_node=" + str(self.config["opt"]["num_gpus"]),
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=127.0.0.1",
            "--master_port=12581",
            "train.py",
            "configs/conf-" + self.id,
        ]
        exec_ext_cmd(cmd, cwd="bin/arcface")

    def training_cleanup(self):
        os.makedirs("arcface_models", exist_ok=True)
        shutil.copy("bin/arcface/work_dirs/conf-{}/model.pt".format(self.id), "arcface_models/" + self.trainingset + ".pt")

        shutil.rmtree("bin/arcface/work_dirs/conf-{}".format(self.id))
        shutil.rmtree("bin/arcface/training-{}".format(self.id))
        os.remove("bin/arcface/configs/conf-" + self.id + ".py")

    @torch.no_grad()
    def load_model(self):
        from bin.arcface.backbones import get_model

        net = get_model("r50", fp16=False)
        net.load_state_dict(torch.load("arcface_models/" + self.trainingset + ".pt"))
        net.eval()
        return net

    @torch.no_grad()
    def load_img(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (112, 112))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        return img

    def cleanup(self):
        del self.net
        del self.reprs
        torch.cuda.empty_cache()
