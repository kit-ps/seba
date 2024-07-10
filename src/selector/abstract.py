import random
import logging


class AbstractSelector:
    name = "abstract"
    random = False

    def __init__(self, config):
        self.log = logging.getLogger("seba.selector")
        self.config = config
        if self.random:
            if "seed" not in self.config:
                raise AttributeError("Random Selectors require seed to be set in configuration!")
            random.seed(a=self.config["seed"])

    def run(self, set):
        new_set = self.select(set)
        new_set.meta["selector"] = self.name
        new_set.meta["params"] = self.config
        new_set.meta["random"] = 0 if not self.random else self.config["seed"]
        new_set.save_meta()
        return new_set

    def select(self, set):
        pass

    def set_train_set(self, set):
        pass
