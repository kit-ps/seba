from .abstract import AbstractExperiment
from ..data.manager import DatasetManager
from ..module_loader import ModuleLoader


class AbstractAnonExperiment(AbstractExperiment):
    def get_anonymization(self):
        if "anonbg_rate" in self.config and self.config["anonbg_rate"] > 0.0:
            self.sets["anon"] = self.get_anonymization_with_bg()
        else:
            self.sets["anon"] = self.get_anonymization_with_parent(self.sets["orig"])

    def get_anonymization_with_parent(self, parent, bg=None):
        config = self.config["anonymization"]
        params = {
            "original": parent.name,
            "anonymization": config["name"],
            "params": {key: val for key, val in config["params"].items() if key != "opt"},
        }
        if bg is not None:
            params["background"] = bg.name
        set = DatasetManager.get_matching(params)
        if set is not None:
            self.log.info("Anonymization skipped. Using existing dataset " + set.name)
            return set
        else:
            return self.run_anonymization(parent, bg)

    def run_anonymization(self, parent, bg):
        config = self.config["anonymization"]
        new_set = parent.copy()
        try:
            anon = ModuleLoader.get_anonymization_by_name(config["name"], self.trait)(config["params"], new_set)
            if bg is not None:
                anon.add_bg(bg)
            anon.run()
        except Exception:
            if self.config["cleanup"]:
                new_set.delete()
            raise RuntimeError("Failed to run anonymization!")
        return new_set

    def get_anonymization_with_bg(self):
        params = {
            "original": self.sets["orig"].name,
            "splitter": "anonbg",
            "params": {"rate": self.config["anonbg_rate"], "seed": 0},
            "part": 0,
        }
        bg = DatasetManager.get_matching(params)
        params["part"] = 1
        parent = DatasetManager.get_matching(params)

        if bg is None or parent is None:
            bg, parent = self.run_anonbg_split()

        self.sets["eval"] = parent
        self.sets["anonbg"] = bg

        return self.get_anonymization_with_parent(parent, bg=bg)

    def run_anonbg_split(self):
        splitter = ModuleLoader.get_splitter_by_name("anonbg")({"rate": self.config["anonbg_rate"], "seed": 0})
        return splitter.run([self.sets["orig"]])
