from ..data.set import Dataset
from ..data.manager import DatasetManager
from ..module_loader import ModuleLoader
from .abstractanon import AbstractAnonExperiment

import copy


class DeanonExperiment(AbstractAnonExperiment):
    def run(self):
        self.sets["orig"] = Dataset(self.config["dataset"])
        self.trait = self.sets["orig"].meta["trait"]
        self.get_anonymization()

        self.get_deanonymization()
        self.run_evaluation()
        self.run_metrics()
        self.save_metrics()

    def get_deanonymization(self):
        self.get_split_sets()
        self.get_deanonymized_set()

    def get_split_sets(self):
        eval_set = self.sets["eval"] if "eval" in self.sets else self.sets["orig"]
        config = self.config["splitter"]
        params = {"original": eval_set.name + "|" + self.sets["anon"].name, "splitter": config["name"], "params": config["params"]}
        sets = []

        params["part"] = 0
        sets.append(DatasetManager.get_matching(params))

        p = copy.deepcopy(params)
        p["params"].pop("parrot", None)
        p["part"] = 1
        sets.append(DatasetManager.get_matching(p))

        p["params"].pop("dist", None)
        p["params"].pop("enroll_rate", None)
        for part in range(2, 4):
            p["part"] = part
            sets.append(DatasetManager.get_matching(p))

        if None in sets:
            splitter = ModuleLoader.get_splitter_by_name(config["name"])(config["params"])
            sets = splitter.run([eval_set, self.sets["anon"]])
        else:
            self.log.info("Splitting skipped. Using existing datasets " + ", ".join(map(lambda x: x.name, sets)))
        self.sets["enroll"] = sets[0]
        self.sets["anon_test"] = sets[1]
        self.sets["clear_train"] = sets[2]
        self.sets["anon_train"] = sets[3]

    def get_deanonymized_set(self):
        config = self.config["deanonymization"]
        params = {
            "original": self.sets["anon_test"].name,
            "deanonymization": config["name"],
            "params": {key: val for key, val in config["params"].items() if key != "opt"},
        }
        new_set = DatasetManager.get_matching(params)
        if new_set is None:
            new_set = self.sets["anon_test"].copy()
            try:
                deanonymization = ModuleLoader.get_deanonymization_by_name(config["name"], self.trait)(config["params"])
                deanonymization.train(self.sets["clear_train"], self.sets["anon_train"])
                deanonymization.run(new_set)
                deanonymization.cleanup()
                del deanonymization
            except Exception:
                if self.config["cleanup"]:
                    new_set.delete()
                raise RuntimeError("Failed to run de-anonymization!")
        else:
            self.log.info("Deanonymization skipped. Using existing dataset " + new_set.name)
        self.sets["test"] = new_set
