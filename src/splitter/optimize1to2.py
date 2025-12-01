from .abstract import AbstractSplitter

import random


class Optimize1to2Splitter(AbstractSplitter):
    """Simple splitter for within a label from an original to an enroll & validation set, leaving points for a future test set

        | enroll | validation | ___ |
              rates[0]     rates[1]

    Required pips:
        none

    Parameters:
        - ([float]) rates: two rates [0, 1] of points per identity to be in the enrollment set and validation set (rest test) (required)
    """

    name = "optimize1to2"
    random = True
    nin = 1
    nout = 2

    def validate_config(self):
        if "rates" not in self.config:
            raise AttributeError("Splitter: config: Missing rate")
        else:
            self.config["rates"][0] = float(self.config["rates"][0])
            self.config["rates"][1] = float(self.config["rates"][1])
            if sum(self.config["rates"]) >= 1:
                raise AttributeError("Splitter: config: rates sum must be lower than 1")

    def split(self, in_sets):
        enroll_points = []
        validation_points = []

        for label in set(map(lambda x: x.label, in_sets[0][:])):
            points = list(filter(lambda x: x.label == label, in_sets[0][:]))
            random.shuffle(points)
            split0 = int(self.config["rates"][0] * len(points))
            split1 = int(self.config["rates"][1] * len(points))
            enroll_points += points[:split0]
            validation_points += points[split0:(split0+split1)]

        enroll_set = in_sets[0].copy(point_filter=(lambda x: x in enroll_points), softlinked=True)
        validation_set = in_sets[0].copy(point_filter=(lambda x: x in validation_points), softlinked=True)

        return [enroll_set, validation_set]
