import logging
import uuid


class AbstractAnonymization:
    name = "abstract"
    random = False

    def __init__(self, config, dataset):
        self.log = logging.getLogger("seba.anonymization")
        self.config = config
        self.dataset = dataset
        self.bg = None

        if self.dataset.meta["original"] is True or ("softlinked" in self.dataset.meta and self.dataset.meta["softlinked"] is True):
            raise AttributeError("Can only run anonymization on non-original hardlinked datasets.")

        self.validate_config()
        self.init()

    def validate_config(self):
        pass

    def init(self):
        pass

    def run(self):
        self.log.info("Running anonymization on dataset " + self.dataset.name)
        self.anonymize_all()
        self.save_meta()
        self.log.info("Anonymization successful.")

    def anonymize_all(self):
        for point in self.dataset.datapoints.values():
            self.anonymize(point)

    def anonymize(self, point):
        pass

    def add_bg(self, bg):
        self.bg = bg

    def save_meta(self):
        self.dataset.meta["anonymization"] = self.name
        self.dataset.meta["params"] = self.config
        self.dataset.meta["random"] = 0 if not self.random else int(uuid.uuid4())
        if self.bg is not None:
            self.dataset.meta["background"] = self.bg.name
        self.dataset.save_meta()
