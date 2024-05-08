from .abstract import AbstractSplitter
from ..lib.data.manager import DatasetManager

import random


class DeanonSplitter(AbstractSplitter):
    """Splitter for de-anonymization evaluation experiments
        Creates four output datasets: clear and anon de-anon training, enrollment and test

    > parrot = false
    CLEAR:          | deanon_train | enroll |
    ANON:           | deanon_train |  test  |
                              train_rate

    > parrot  = true
    CLEAR:          | deanon_train |               |
    ANON:           | deanon_train |  enroll/test  |
                              train_rate
    Required pips:
        none

    Parameters:
        - (string) seed: specify a custom seed to use when perturbating image ids (optional, random default)
        - (float) train_rate: rate [0, 1] of identities to include in the training sets.
        - (float) enroll_rate: rate [0, 1] of datapoints per identity (which are not in the train set) to include in the enroll set
        - (bool) parrot: enroll set uses anonymized images instead of clear images (optional, default: false)
    """

    name = "deanon"
    random = True
    nin = 2
    nout = 4

    def validate_config(self):
        if "parrot" not in self.config:
            self.config["parrot"] = False

        if "enroll_rate" not in self.config:
            raise AttributeError("Splitter: config: Missing enroll_rate")
        else:
            self.config["enroll_rate"] = float(self.config["enroll_rate"])
            if self.config["enroll_rate"] <= 0 or self.config["enroll_rate"] >= 1:
                raise AttributeError("Splitter: config: enroll_rate not in [0,1]")

        if "train_rate" not in self.config:
            raise AttributeError("Splitter: config: Missing train_rate")
        else:
            self.config["train_rate"] = float(self.config["train_rate"])
            if self.config["train_rate"] < 0 or self.config["train_rate"] >= 1:
                raise AttributeError("Splitter: config: train_rate not in [0,1]")

    def split(self, in_sets):
        orig_set, anon_set = in_sets
        ids = list(orig_set.identities.keys())
        random.shuffle(ids)

        split = int(len(ids) * self.config["train_rate"])

        deanon_ids = ids[:split]
        test_ids = ids[split:]

        clear_train_set, anon_train_set = self.create_train(orig_set, anon_set, deanon_ids)
        enroll_set, anon_test_set = self.create_test(orig_set, anon_set, test_ids)

        return [enroll_set, anon_test_set, clear_train_set, anon_train_set]

    def create_train(self, orig_set, anon_set, deanon_ids):
        clear_train_set, anon_train_set = self.existing_train(orig_set.name, anon_set.name)
        if clear_train_set is None or anon_train_set is None:
            self.log.info("Creating training data set (clear)...")
            clear_train_set = orig_set.copy(only_ids=deanon_ids, only_points=False, softlinked=True)
            self.log.info("Creating training data set (anon)...")
            anon_train_set = anon_set.copy(only_ids=deanon_ids, only_points=False, softlinked=True)
        else:
            self.log.info("Re-Using training data sets: {} (clear) & {} (anon)".format(clear_train_set.name, anon_train_set.name))
        return clear_train_set, anon_train_set

    def create_test(self, orig_set, anon_set, test_ids):
        enroll_img_ids = []
        test_img_ids = []

        for identity in orig_set.point_by_id():
            random.shuffle(identity)
            split = int(self.config["enroll_rate"] * len(identity))
            enroll_img_ids += identity[:split]
            test_img_ids += identity[split:]

        if self.config["parrot"]:
            self.log.info("Creating enroll data set (copy of anon)...")
            enroll_set = anon_set.copy(only_ids=test_ids, only_points=enroll_img_ids, softlinked=True)
        else:
            self.log.info("Creating enroll data set (copy of clear)...")
            enroll_set = orig_set.copy(only_ids=test_ids, only_points=enroll_img_ids, softlinked=True)

        anon_test_set = self.existing_test(orig_set.name, anon_set.name)
        if anon_test_set is None:
            self.log.info("Creating test data set...")
            anon_test_set = anon_set.copy(only_ids=test_ids, only_points=test_img_ids, softlinked=True)
        else:
            self.log.info("Re-Using test data set {}".format(anon_test_set.name))
        return enroll_set, anon_test_set

    def existing_train(self, orig_set_name, anon_set_name):
        # params that do *not* influence the training sets: parrot, enroll_rate
        # params that *do* influence the training sets: train_rate, seed, parent sets
        config = {
            "train_rate": self.config["train_rate"],
            "seed": self.config["seed"],
        }
        params = {"original": orig_set_name + "|" + anon_set_name, "splitter": self.name, "params": config, "part": 2}
        clear_set = DatasetManager.get_matching(params)
        if clear_set is not None:
            clear_set.meta["params"] = config
            clear_set.save_meta()

        params["part"] = 3
        anon_set = DatasetManager.get_matching(params)
        if anon_set is not None:
            anon_set.meta["params"] = config
            anon_set.save_meta()

        return clear_set, anon_set

    def existing_test(self, orig_set_name, anon_set_name):
        # params that do *not* influence the training sets: parrot
        # params that *do* influence the training sets: train_rate, seed, parent sets, enroll_rate
        config = {"train_rate": self.config["train_rate"], "seed": self.config["seed"], "enroll_rate": self.config["enroll_rate"]}
        params = {"original": orig_set_name + "|" + anon_set_name, "splitter": self.name, "params": config, "part": 1}
        test_set = DatasetManager.get_matching(params)
        if test_set is not None:
            test_set.meta["params"] = config
            test_set.save_meta()

        return test_set
