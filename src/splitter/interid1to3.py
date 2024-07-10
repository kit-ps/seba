from .abstract import AbstractSplitter

import random


class Interid1to3Splitter(AbstractSplitter):
    """Simple randomized splitter for anonymization background sets
        Creates three output data sets: background, attacker and evaluation

        | anon_bg | attacker | evaluation |
                rate_0     rate_1

    Required pips:
        none

    Parameters:
        - (list[float]) rates: two rates [0, 1] of datapoints to include in the anon background set and attacker dataset, respectively
    """

    name = "interid1to3"
    random = True
    nin = 1
    nout = 3

    def validate_config(self):
        if "rates" not in self.config:
            raise AttributeError("Splitter: config: Missing rates")
        else:
            self.config["rates"] = list(map(lambda x: float(x), self.config["rates"]))
            if len(self.config["rates"]) != 2:
                raise AttributeError("Splitter: Expected exactly two rates!")
            if self.config["rates"][0] < 0 or self.config["rates"][1] < 0 or sum(self.config["rates"]) > 1:
                raise AttributeError("Splitter: config: rate not in [0,1]")

    def split(self, in_sets):
        in_set = in_sets[0]
        ids = list(in_set.identities.keys())
        random.shuffle(ids)

        split0 = int(len(ids) * self.config["rates"][0])
        split1 = split0 + int(len(ids) * self.config["rates"][1])

        anonbg_ids = ids[:split0]
        attacker_ids = ids[split0:split1]
        eval_ids = ids[split1:]

        self.log.info("Creating anonymization background set...")
        anonbg_set = in_set.copy(only_ids=anonbg_ids, softlinked=True)

        self.log.info("Creating attacker set...")
        attacker_set = in_set.copy(only_ids=attacker_ids, softlinked=True)

        self.log.info("Creating evaluation set...")
        eval_set = in_set.copy(only_ids=eval_ids, softlinked=True)

        return [anonbg_set, attacker_set, eval_set]
