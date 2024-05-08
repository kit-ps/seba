# Process a dataset to add classification accuracies, encodings and zoo-metrics.

import argparse
import json
import os
import logging
import sys
import time
from tqdm import tqdm
import numpy as np
from deepface.commons.distance import findCosineDistance, findEuclideanDistance

from src.lib.data.set import Dataset
from src.lib.module_loader import ModuleLoader


def fullset_recognition(recognition, dataset, training_set):
    splitter = ModuleLoader.get_splitter_by_name("anon")({"enroll_anon": False, "test_anon": False, "rate": 0.75})
    enroll, test = splitter.run([dataset, dataset])
    if training_set:
        ts = Dataset(training_set)
        recognition.train(ts)
    recognition.enroll(enroll)
    return recognition.recognize(test, False)


def calcandwrite_idlevelacc(results, dataset):
    metric = ModuleLoader.get_metric_by_name("idlevel")(results)
    metric.calc()

    for id in metric.data:
        d = {}
        for k, v in zip(metric.keys, id):
            if k not in ["identity", "n"]:
                d["classification_" + k] = float(v)
        dataset.identities[id[0]].save_attr_batch(d)


def save_encodings(recognition, dataset):
    for point in dataset.datapoints.values():
        enc = recognition.get_encoding(point)
        enc = json.loads(json.dumps(enc))
        point.save_attr("embedding", enc)


def zoo_metrics(dataset):
    pbi = dataset.point_by_id()

    for id in tqdm(pbi):
        identity = dataset.identities[dataset.datapoints[id[0]].idname]
        embds = list(map(lambda x: dataset.datapoints[x].embedding, id))

        # CENTER
        center = json.loads(json.dumps(np.mean(np.array(embds), axis=0).tolist()))
        identity.save_attr("embed_center", center)

        # GENUINE
        dsts_cos = []
        dsts_eucl = []

        for x in embds:
            dsts_cos.append(findCosineDistance(center, x))
            dsts_eucl.append(findEuclideanDistance(center, x))

        identity.save_attr(
            "gen_score_cos", json.loads(json.dumps({"min": min(dsts_cos), "max": max(dsts_cos), "avg": sum(dsts_cos) / len(dsts_cos)}))
        )
        identity.save_attr(
            "gen_score_eucl", json.loads(json.dumps({"min": min(dsts_eucl), "max": max(dsts_eucl), "avg": sum(dsts_eucl) / len(dsts_eucl)}))
        )

    # IMPOSTER
    for id in tqdm(pbi):
        identity = dataset.identities[dataset.datapoints[id[0]].idname]
        center = identity.embed_center

        dsts_cos = []
        dsts_eucl = []

        for point in dataset.datapoints.values():
            if point.idname is not identity.name:
                dsts_cos.append(findCosineDistance(center, point.embedding))
                dsts_eucl.append(findEuclideanDistance(center, point.embedding))

        identity.save_attr(
            "imp_score_cos", json.loads(json.dumps({"min": min(dsts_cos), "max": max(dsts_cos), "avg": sum(dsts_cos) / len(dsts_cos)}))
        )
        identity.save_attr(
            "imp_score_eucl", json.loads(json.dumps({"min": min(dsts_eucl), "max": max(dsts_eucl), "avg": sum(dsts_eucl) / len(dsts_eucl)}))
        )


os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("logs/" + time.strftime("%Y-%m-%d_%H-%M") + ".log"), logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("seba")

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", default="{}", help="Paramters for the recognition method.")
parser.add_argument("-t", "--training_set", help="Dataset to train recognition with", required=False)
parser.add_argument("dataset", help="The name of the anonymization method to use.")
parser.add_argument("recognition", help="The data set to anonymize")
args = parser.parse_args()

logger.info("Loading dataset and recognition method")
dataset = Dataset(args.dataset)
recognition = ModuleLoader.get_recognition_by_name(args.recognition, dataset.meta["trait"])(json.loads(args.params))

logger.info("Performing full set recognition")
results = fullset_recognition(recognition, dataset, args.training_set)

logger.info("Calculating idlevel accuracy metrics and writing to metadata")
calcandwrite_idlevelacc(results, dataset)

logger.info("Extracting encodings for all datapoints")
save_encodings(recognition, dataset)

logger.info("Calculating zoo metrics")
zoo_metrics(dataset)

logger.info("Success.")
