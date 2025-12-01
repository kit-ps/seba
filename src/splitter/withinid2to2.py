from .abstract import AbstractSplitter

import random


class Withinid2to2Splitter(AbstractSplitter):
    """Splitter for anonymization evaluation experiments
        Creates two output datasets: enrollment and test

        | enroll | test |
                rate

    Required pips:
        none

    Parameters:
        - (float) rate: rate [0, 1] of images per identity to be in the enrollment set (rest test) (1.0 for comparison)(required)
        - (bool) enroll_clear: whether enrollment images are from clear set (true) or anonymized set (false) (optional)
    """

    name = "withinid2to2"
    random = True
    nin = 2
    nout = 2

    def validate_config(self):
        if "rate" not in self.config:
            raise AttributeError("Splitter: config: Missing rate")
        else:
            self.config["rate"] = float(self.config["rate"])
            if self.config["rate"] < 0 or self.config["rate"] > 1:
                raise AttributeError("Splitter: config: rate not in [0,1]")

        if "enroll_clear" not in self.config:
            self.config["enroll_clear"] = False
        else:
            self.config["enroll_clear"] = bool(self.config["enroll_clear"])

    def split(self, in_sets):
        orig_set, anon_set = in_sets

        enroll_points = []
        test_points = []

        min_set = anon_set if len(anon_set.identities) <= len(orig_set.identities) else orig_set

        if self.config["rate"] == 1.0:
            return [
                orig_set.copy(id_filter=(lambda x: x.identity in min_set.identities), softlinked=True),
                anon_set.copy(id_filter=(lambda x: x.idenitty in min_set.identities), softlinked=True),
            ]

        for identity in min_set.identities:
            imgs = min_set[identity]
            random.shuffle(imgs)
            split = int(self.config["rate"] * len(imgs))
            enroll_points += imgs[:split]
            test_points += imgs[split:]

        if self.config["enroll_clear"]:
            enroll_set = orig_set.copy(point_filter=(lambda x: x in enroll_points), softlinked=True)
        else:
            enroll_set = anon_set.copy(point_filter=(lambda x: x in enroll_points), softlinked=True)
        test_set = anon_set.copy(point_filter=(lambda x: x in test_points), softlinked=True)

        return [enroll_set, test_set]
