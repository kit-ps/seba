import logging
import random

class AbstractDeanonymization:
    name = "abstract"
    nout = 1

    def __init__(self, config, seed):
        self.log = logging.getLogger("seba.deanonymization")
        self.config = config
        self.seed = seed
        random.seed(a=self.seed)
        self.bg = None

        self.validate_config()
        self.init()

    def validate_config(self):
        pass

    def init(self):
        pass

    def train(self, clear_set, anon_set):
        pass

    def run(self, datasets):
        self.clear_train, self.anon_train, self.parent = datasets
        self.train(self.clear_train, self.anon_train)
        self.dataset = self.parent.copy(softlinked=False)
        self.log.info("Running deanonymization on dataset " + self.dataset.name)
        self.deanonymize_all()
        self.save_meta()
        self.log.info("Deanonymization successful.")

    def deanonymize_all(self):
        for point in self.dataset[:]:
            self.deanonymize(point)

    def deanonymize(self, point):
        pass

    def save_meta(self):
        self.dataset.meta["original"] = self.clear_train.name + "|" + self.anon_train.name + "|" + self.parent.name
        self.dataset.meta["type"] = "deanonymization"
        self.dataset.meta["name"] = self.name
        self.dataset.meta["params"] = self.config
        self.dataset.meta["seed"] = self.seed
        self.dataset.meta["part"] = 0

        self.dataset.save_meta()

    def cleanup(self):
        pass
