from .utils import recursive_replace
from .module_loader import ModuleLoader

import logging
import os


class Run:
    def __init__(self, config, round=0, save_result=False):
        config = recursive_replace(config, "$ROUND", round)
        self.log = logging.getLogger("seba.run")
        self.log.info("=" * 30)
        self.log.info("Starting new run.\n\tConfiguration: " + str(config))
        self.config = config
        self.sets = {}
        self.save_result = save_result
        if "cleanup" not in self.config:
            self.config["cleanup"] = False
        os.makedirs("results", exist_ok=True)

    def run(self):
        exp = ModuleLoader.get_exp_by_name(self.config["exp"])(self.config, self.save_result)
        exp.run()
