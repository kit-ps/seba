from .abstract import AbstractSplitter

import random


class AnonbgSplitter(AbstractSplitter):
    """Simple randomized splitter for anonymization background sets
        Creates two output data sets: background and evaluation

        | anon_bg | eval |
                 rate

    Required pips:
        none

    Parameters:
        - (float) rate: rate [0, 1] of datapoints to include in the anon background set
        - (string) seed: specify a custom seed to use when perturbating image ids (optional, random default)
    """

    name = "anonbg"
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
        in_set = in_sets[0]
        ids = list(in_set.identities.keys())
        random.shuffle(ids)

        split = int(len(ids) * self.config["rate"])

        anonbg_ids = ids[:split]
        eval_ids = ids[split:]

        self.log.info("Creating anonymization background set...")
        anonbg_set = in_set.copy(only_ids=anonbg_ids, softlinked=True)

        self.log.info("Creating evaluation set...")
        eval_set = in_set.copy(only_ids=eval_ids, softlinked=True)

        return [anonbg_set, eval_set]
