from .abstract import AbstractSplitter

import random


class Withinlabel1to2Splitter(AbstractSplitter):
    """Simple splitter for within a label from an original to an enroll & test set.

        | enroll | test |
                rate

    Required pips:
        none

    Parameters:
        - (float) rate: rate [0, 1] of images per identity to be in the enrollment set (rest test) (required)
    """

    name = "withinlabel1to2"
    random = True
    nin = 1
    nout = 2

    def validate_config(self):
        if "rate" not in self.config:
            raise AttributeError("Splitter: config: Missing rate")
        else:
            self.config["rate"] = float(self.config["rate"])
            if self.config["rate"] < 0 or self.config["rate"] > 1:
                raise AttributeError("Splitter: config: rate not in [0,1]")

    def split(self, in_sets):
        enroll_points = []
        test_points = []

        for label in set(map(lambda x: x.label, in_sets[0][:])):
            points = list(filter(lambda x: x.label == label, in_sets[0][:]))
            random.shuffle(points)
            split = int(self.config["rate"] * len(points))
            enroll_points += points[:split]
            test_points += points[split:]

        enroll_set = in_sets[0].copy(point_filter=(lambda x: x in enroll_points), softlinked=True)
        test_set = in_sets[0].copy(point_filter=(lambda x: x in test_points), softlinked=True)

        return [enroll_set, test_set]
