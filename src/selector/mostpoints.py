from .abstract import AbstractSelector

import random


class MostpointsSelector(AbstractSelector):
    """Use the identities with highest number of datapoints in the set
        Between identities with the same number of datapoints, choose randomly.

    Required pips:
        none

    Parameters:
        - (int) ids: number of identities to select
        - (int) offset: number of identities to skip at beginning (optional, default: 0)
    """

    name = "mostpoints"
    random = True

    def select(self, set):
        ids = list(set.identities.items())
        random.shuffle(ids)
        ids.sort(key=lambda x: x[1].npoints, reverse=True)

        offset = 0
        if "offset" in self.config:
            offset = int(self.config["offset"])

        ids = list(map(lambda x: x[0], ids))
        only_ids = ids[offset : (offset + self.config["ids"])]

        return set.copy(only_ids=only_ids, softlinked=True)
