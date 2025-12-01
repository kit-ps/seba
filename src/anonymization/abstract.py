import logging
import random

class AbstractAnonymization:
    name = "abstract"
    random = False
    nout = 1

    def __init__(self, config, seed):
        self.log = logging.getLogger("seba.anonymization")
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

    def run(self, datasets):
        if len(datasets) > 1:
            self.parent, self.bg = datasets
        else:
            self.parent = datasets[0]
        self.dataset = self.parent.copy(softlinked=False)
        self.log.info("Running anonymization on dataset " + self.dataset.name)
        self.anonymize_all()
        self.save_meta()
        self.log.info("Anonymization successful.")
        return self.dataset

    def anonymize_all(self):
        for point in self.dataset[:]:
            self.anonymize(point)

    def anonymize(self, point):
        pass

    def save_meta(self):
        if self.bg is None:
            self.dataset.meta["original"] = self.parent.name
        else:
            self.dataset.meta["original"] = self.parent.name + "|" + self.bg.name
        self.dataset.meta["type"] = "anonymization"
        self.dataset.meta["name"] = self.name
        self.dataset.meta["params"] = self.config
        self.dataset.meta["seed"] = self.seed
        self.dataset.meta["part"] = 0

        self.dataset.save_meta()
