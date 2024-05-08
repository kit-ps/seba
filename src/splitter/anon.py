from .abstract import AbstractSplitter

import random


class AnonSplitter(AbstractSplitter):
    """Splitter for anonymization evaluation experiments
        Creates two output datasets: enrollment and test

        | enroll | test |
                rate

    Required pips:
        none

    Parameters:
        - (bool) enroll_anon: datapoints in enrollment set are anonymized (optional, default false)
        - (bool) test_anon: datapoints in test set are anonymized (optional, default true)
        - (float) rate: rate [0, 1] of images per identity to be in the enrollment set (rest test) (required)
    """

    name = "anon"
    random = True
    nin = 2
    nout = 2

    def validate_config(self):
        if "enroll_anon" not in self.config:
            self.config["enroll_anon"] = False

        if "test_anon" not in self.config:
            self.config["test_anon"] = True

        if "rate" not in self.config:
            raise AttributeError("Splitter: config: Missing rate")
        else:
            self.config["rate"] = float(self.config["rate"])
            if self.config["rate"] < 0 or self.config["rate"] > 1:
                raise AttributeError("Splitter: config: rate not in [0,1]")

    def split(self, in_sets):
        orig_set, anon_set = in_sets

        enroll_img_ids = []
        test_img_ids = []

        min_set = anon_set if len(anon_set.identities) <= len(orig_set.identities) else orig_set

        for identity in min_set.point_by_id():
            random.shuffle(identity)
            split = int(self.config["rate"] * len(identity))
            enroll_img_ids += identity[:split]
            test_img_ids += identity[split:]

        if self.config["enroll_anon"]:
            enroll_set = anon_set.copy(only_points=enroll_img_ids, softlinked=True)
        else:
            enroll_set = orig_set.copy(only_points=enroll_img_ids, softlinked=True)

        if self.config["test_anon"]:
            test_set = anon_set.copy(only_points=test_img_ids, softlinked=True)
        else:
            test_set = orig_set.copy(only_points=test_img_ids, softlinked=True)

        return [enroll_set, test_set]
