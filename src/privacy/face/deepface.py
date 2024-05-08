from ...lib.result import Result
from .abstract import AbstractFacePrivacy
from ...lib.inference import Classification
from ...lib.utils import suppress_stdout

from deepface import DeepFace


class DeepfaceRecognition(Classification, AbstractFacePrivacy):
    """Use the deepface privacy method.
    DeepFace documentation: https://github.com/serengil/deepface

    Required pips:
        - deepface==0.0.65

    NOTE: This was tested with deepface=0.0.65 and we recommend using this version.
    v66 and v67 seem to be bugged when using retinaface detector and starting with v68 privacy performance significantly decreased.
    However, with the recommended version, the DeepID model is bugged when using tensorflow >= 2.5
    - to fix this issue apply the patch from scripts/installer/deepface_dlib_0065.patch to your DeepFace installation.

    NOTE: You may want to apply the patch from scripts/installer/deepface_threshold.patch to save all match values.

    NOTE: To be able to use the already_normalized option, you will need to patch your DeepFace installation.
    See scripts/installer/deepface_normalization.patch

    Parameters:
        - detector (string): detector backend for face detection, one of retinaface, mtcnn, opencv, ssd or dlib
        - model (string): model name for face privacy, one of VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib or Ensemble
        - distance (string): distance metric for face privacy, one of cosine, euclidean, euclidean_l2
        - already_normalized (bool): whether images in the dataset have already been cropped and aligned (recommended for performance)
    """

    def validate_config(self):
        if "detector" not in self.config:
            self.config["detector"] = "retinaface"

        if "model" not in self.config:
            self.config["model"] = "VGG-Face"

        if "distance" not in self.config:
            self.config["distance"] = "cosine"

        if "already_normalized" not in self.config:
            self.config["already_normalized"] = False
        else:
            self.config["already_normalized"] = bool(self.config["already_normalized"])

    def enroll(self, set):
        self.folder = set.folder
        self.log.info("Starting privacy evaluation.\n\tEnroll-Folder: " + self.folder + "\n\tConfiguration: " + str(self.config))

    def classify_all(self, set, results):
        with suppress_stdout():
            result_p = DeepFace.find(
                list(map(lambda x: x.get_path(), set.datapoints.values())),
                self.folder,
                model_name=self.config["model"],
                enforce_detection=False,
                detector_backend=self.config["detector"],
                distance_metric=self.config["distance"],
                prog_bar=False,
                already_normalized=self.config["already_normalized"],
            )

        i = 0
        for name, point in set.datapoints.items():
            results.append(self.classify_point(name, point, result_p[i].to_dict()))
            i += 1
        return results

    def classify_point(self, name, point, result):
        dist_key = self.config["model"] + "_" + self.config["distance"]
        rs = Result(point.idname, point.pointname)

        for i in range(len(result["identity"])):
            dist = round(result[dist_key][i], 3)
            rs.add_recognized(result["identity"][i].split("/")[-1].split(".")[0], dist=dist)
        self.log.debug(str(rs))
        return rs

    def get_encoding(self, point):
        with suppress_stdout:
            return DeepFace.represent(
                img_path=point.get_path(),
                model_name=self.config["model"],
                detector_backend=self.config["detector"],
                already_normalized=self.config["already_normalized"],
            )
