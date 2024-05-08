from ..data.set import Dataset

import copy
import logging
import yaml
import json


class AbstractExperiment:
    def __init__(self, config, save_result):
        self.log = logging.getLogger("seba.exp")
        self.config = config
        self.save_result = save_result
        self.sets = {}
        self.orig_config = copy.deepcopy(config)

    def run_evaluation(self):
        if "privacy" in self.config:
            self.run_recognition()
        elif "utility" in self.config:
            self.run_utility()

    def run_recognition(self):
        from ..module_loader import ModuleLoader

        rec_module = ModuleLoader.get_recognition_by_name(self.config["privacy"]["name"], self.trait)
        recognition = rec_module(self.config["privacy"]["params"])
        self.metrics = recognition.metrics

        if "train_set" in self.config:
            recognition.train(Dataset(self.config["train_set"]))

        self.resultset = recognition.run(self.sets["enroll"], self.sets["test"], self.save_result)
        self.resultset.save_context(self.orig_config, dict(map(lambda x: (x[0], x[1].name), self.sets.items())))
        recognition.cleanup()

    def run_utility(self):
        from ..module_loader import ModuleLoader

        ut_module = ModuleLoader.get_utility_by_name(self.config["utility"]["name"], self.trait)
        utility = ut_module(self.config["utility"]["params"])
        self.metrics = utility.metrics

        if "train_set" in self.config:
            utility.train(Dataset(self.config["train_set"]))

        self.resultset = utility.run(self.sets["enroll"], self.sets["test"], self.save_result)
        self.resultset.save_context(self.orig_config, dict(map(lambda x: (x[0], x[1].name), self.sets.items())))

    def run_metrics(self):
        from ..module_loader import ModuleLoader

        self.rs = {}
        for metric in self.metrics:
            m = ModuleLoader.get_metric_by_name(metric)(self.resultset)
            r = m.run()
            for k, v in r.items():
                if k not in self.rs:
                    self.rs[k] = v

    def save_metrics(self):
        with open("results.yaml", "a") as f:
            sets = dict(map(lambda x: (x[0], x[1].name), self.sets.items()))
            z = json.loads(json.dumps({self.resultset.id: {"metrics": self.rs, "config": self.orig_config, "datasets": sets}}))
            f.write(yaml.dump(z))
