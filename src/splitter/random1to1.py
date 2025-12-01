from .abstract import AbstractSplitter

import random


class Random1to1Splitter(AbstractSplitter):
    """Use random identities in the set

    Required pips:
        none

    Parameters:
        - (int) ids: number of identities to select
    """

    name = "random1to1"
    random = True
    nin = 1
    nout = 1

    def validate_config(self):
        if "ids" not in self.config:
            raise AttributeError("Splitter: config: Missing number of ids")

    def split(self, in_sets):
        ids = in_sets[0].identities
        random.shuffle(ids)
        id_filter = (lambda x: x.identity in ids[: self.config["ids"]])

        return [in_sets[0].copy(id_filter=id_filter, softlinked=True)]
