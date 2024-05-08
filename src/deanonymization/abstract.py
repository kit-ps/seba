import logging


class AbstractDeanonymization:
    name = "abstract"

    def __init__(self, config):
        self.log = logging.getLogger("seba.deanonymization")
        self.config = config

        self.validate_config()
        self.init()

    def validate_config(self):
        pass

    def init(self):
        pass

    def train(self, clear_set, anon_set):
        pass

    def run(self, dataset):
        self.dataset = dataset
        if self.dataset.meta["original"] is True or ("softlinked" in self.dataset.meta and self.dataset.meta["softlinked"] is True):
            raise AttributeError("Can only run deanonymization on non-original hardlinked datasets.")

        self.log.info("Running deanonymization on dataset " + self.dataset.name)
        self.deanonymize_all()
        self.save_meta()
        self.log.info("Deanonymization successful.")

    def deanonymize_all(self):
        for point in self.dataset.datapoints.values():
            self.deanonymize(point)

    def deanonymize(self, point):
        pass

    def save_meta(self):
        self.dataset.meta["deanonymization"] = self.name
        self.dataset.meta["params"] = self.config
        self.dataset.save_meta()

    def cleanup(self):
        pass
