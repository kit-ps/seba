import logging
import random

class AbstractProcessing:
    name = "abstract"
    random = False
    softlinked = False
    nout = 1

    def __init__(self, config, seed, context):
        self.log = logging.getLogger("seba.processing")
        self.config = config
        self.seed = seed
        self.context = context
        random.seed(a=self.seed)
        self.context = context

        self.validate_config()
        self.init()

    def validate_config(self):
        pass

    def init(self):
        pass

    def run(self, datasets):
        self.parent = datasets[0]
        self.dataset = self.parent.copy(softlinked=self.softlinked)
        self.log.info("Running processing on dataset " + self.dataset.name)
        self.process_all()
        self.save_meta()
        self.log.info("Processing successful.")
        return self.dataset

    def process_all(self):
        for point in self.dataset[:]:
            self.process(point)

    def process(self, point):
        pass

    def save_meta(self):
        self.dataset.meta["original"] = self.parent.name
        self.dataset.meta["type"] = "processing"
        self.dataset.meta["name"] = self.name
        self.dataset.meta["label"] = self.dataset.name
        self.dataset.meta["params"] = self.config
        self.dataset.meta["seed"] = self.seed
        self.dataset.meta["part"] = 0

        self.dataset.save_meta()
        self.dataset.scan()
