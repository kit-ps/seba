from .abstract import AbstractSelector
from bin.arcface.backbones import get_model

from deepface.commons.distance import findEuclideanDistance
import os.path
import torch
import json
import cv2
import numpy as np


class DistinctiveSelector(AbstractSelector):
    """Use the genuine and imposter scores of every identity to sort them

    Required pips:
        none

    Parameters:
        - (int) ids: number of identities to select
        - (int) min_imgs_per_id: minimum number of images per identity.
    """

    name = "distinctive"

    def set_train_set(self, set):
        self.modelname = set.name
        if not os.path.exists("arcface_models/" + self.modelname + ".pt"):
            self.log.info("Model for embeddings for distinctive has not been trained yet.")
            from ..privacy.face.arcface import ArcfaceClassification

            arcface = ArcfaceClassification({"opt": {"num_gpus": 1}})
            arcface.train(set)

    def select(self, set):
        pbi = set.point_by_id()
        if "min_imgs_per_id" in self.config:
            pbi = list(filter(lambda x: len(x) >= int(self.config["min_imgs_per_id"]), pbi))
        scores = {}
        embeds = {}
        ids = []

        self.log.info("Calculating embeddings..")
        with torch.no_grad():
            net = get_model("r50", fp16=False)
            net.load_state_dict(torch.load("arcface_models/" + self.modelname + ".pt"))
            net.eval()

            for key, image in set.datapoints.items():
                img = cv2.imread(image.get_path())
                img = cv2.resize(img, (112, 112))

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.transpose(img, (2, 0, 1))
                img = torch.from_numpy(img).unsqueeze(0).float()
                img.div_(255).sub_(0.5).div_(0.5)

                res = net(img).numpy()[0].tolist()
                res = json.loads(json.dumps(res))
                embeds[key] = res

        self.log.info("Calculating scores..")
        for id in pbi:
            ids.append(set.datapoints[id[0]].idname)
            dsts = []
            id_em = np.array(list(map(lambda x: embeds[x], id)))
            center = json.loads(json.dumps(np.mean(id_em, axis=0).tolist()))

            # Calculate the genuine score for each identity within a dataset
            dsts = []
            for x in id_em:
                dsts.append(json.loads(json.dumps(findEuclideanDistance(center, x))))
            gen_score = {"min": min(dsts), "max": max(dsts), "avg": sum(dsts) / len(dsts)}

            # Calculate the genuine score for each identity within a dataset
            dsts = []
            for key, point in set.datapoints.items():
                if not point.idname is set.datapoints[id[0]].idname:
                    dsts.append(json.loads(json.dumps(findEuclideanDistance(center, embeds[key]))))
            imp_score = {"min": min(dsts), "max": max(dsts), "avg": sum(dsts) / len(dsts)}
            scores[set.datapoints[id[0]].idname] = {"imp_score": imp_score, "gen_score": gen_score}

        self.log.info("Selecting identities..")
        ids = list(map(lambda x: (x, (-1) * scores[x]["gen_score"]["max"] + scores[x]["imp_score"]["min"]), ids))

        ids.sort(key=lambda x: x[1], reverse=True)
        only_ids = list(map(lambda x: x[0], ids))[: self.config["ids"]]
        return set.copy(only_ids=only_ids, softlinked=True)
