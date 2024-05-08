import logging
import random


class AbstractSplitter:
    name = "abstract"
    random = False
    nin = 0
    nout = 0

    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger("seba.splitter")
        self.validate_config()
        if self.random:
            if "seed" not in self.config:
                raise AttributeError("Random Splitters require seed to be set in configuration!")
            random.seed(a=self.config["seed"])

    def validate_config(self):
        pass

    def run(self, in_sets):
        if not len(in_sets) == self.nin:
            raise AttributeError(
                "Splitter: number of input sets must be {} for splitter {}. Received {}.".format(self.nin, self.name, len(in_sets))
            )

        out_sets = self.split(in_sets)

        if not len(out_sets) == self.nout:
            raise RuntimeError(
                "Splitter: internal error, number of output sets does not match splitter specification. Expected {}, received {}.".format(
                    self.nout, len(out_sets)
                )
            )

        parents = "|".join(list(map(lambda x: x.name, in_sets)))

        for i in range(len(out_sets)):
            if not ("part" in out_sets[i].meta and out_sets[i].meta["part"] == i):
                # this allows skipping meta writing here if meta was written within splitter
                out_sets[i].meta["original"] = parents
                out_sets[i].meta["params"] = self.config
                out_sets[i].meta["splitter"] = self.name
                out_sets[i].meta["part"] = i
                out_sets[i].save_meta()

        return out_sets

    def split(self, in_sets):
        return []
