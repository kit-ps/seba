from .abstract import AbstractSelector

import random


class RandomSelector(AbstractSelector):
    """Use random identities in the set

    Required pips:
        none

    Parameters:
        - (int) ids: number of identities to select
        - (int) min_img_per_id: minimum number of datapoints per identity to select (ids with less imgs will not be chosen)
        - (int) max_img_per_id: maximum number of datapoints per identity to select (ids with more img will be chosen but imgs cut off)
    """

    name = "random"
    random = True

    def select(self, set):
        ids = set.identities
        if "min_img_per_id" in self.config:
            ids = dict(filter(lambda x: x[1].npoints >= self.config["min_img_per_id"], ids.items()))

        ids = list(ids.keys())
        random.shuffle(ids)
        only_ids = ids[: self.config["ids"]]

        if "max_img_per_id" in self.config:
            only_imgs = []
            for id in set.point_by_id():
                random.shuffle(id)
                only_imgs += id[: self.config["max_img_per_id"]]
        else:
            only_imgs = False

        return set.copy(only_ids=only_ids, only_points=only_imgs, softlinked=True)
