from ..data.set import Dataset
from ..data.manager import DatasetManager
from ..module_loader import ModuleLoader
from .abstractanon import AbstractAnonExperiment


class AnonExperiment(AbstractAnonExperiment):
    def run(self):
        self.sets["orig"] = Dataset(self.config["dataset"])
        self.trait = self.sets["orig"].meta["trait"]
        self.get_anonymization()
        self.sets["eval"] = self.get_selected_set()

        self.get_splitting()
        self.run_evaluation()
        self.run_metrics()
        self.save_metrics()

    def get_selected_set(self):
        config = self.config["selector"]
        params = {"original": self.sets["anon"].name, "selector": config["name"], "params": config["params"]}
        new_set = DatasetManager.get_matching(params)
        if new_set is None:
            selector = ModuleLoader.get_selector_by_name(config["name"])(config["params"])
            new_set = selector.run(self.sets["anon"])
        else:
            self.log.info("Selection skipped. Using existing dataset " + new_set.name)
        return new_set

    def get_splitting(self):
        config = self.config["splitter"]
        params = {
            "original": self.sets["orig"].name + "|" + self.sets["eval"].name,
            "splitter": config["name"],
            "params": config["params"],
            "part": 0,
        }
        enroll_set = DatasetManager.get_matching(params)
        params["part"] = 1
        test_set = DatasetManager.get_matching(params)
        if enroll_set is None or test_set is None:
            splitter = ModuleLoader.get_splitter_by_name(config["name"])(config["params"])
            enroll_set, test_set = splitter.run([self.sets["orig"], self.sets["eval"]])
        else:
            self.log.info("Splitting skipped. Using existing datasets " + enroll_set.name + " & " + test_set.name)
        self.sets["enroll"] = enroll_set
        self.sets["test"] = test_set
