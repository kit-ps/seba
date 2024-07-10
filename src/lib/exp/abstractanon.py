from .abstract import AbstractExperiment
from ..data.manager import DatasetManager
from ..module_loader import ModuleLoader


class AbstractAnonExperiment(AbstractExperiment):
    def get_anonymized_set(self, parent):
        params = {
            "original": parent.name,
            "anonymization": self.config["anonymization"]["name"],
            "background": self.sets["anonbg"].name,
            "params": {key: val for key, val in self.config["anonymization"]["params"].items() if key != "opt"},
        }
        set = DatasetManager.get_matching(params)
        if set is not None:
            self.log.info("Anonymization skipped. Using existing dataset " + set.name)
            return set
        else:
            return self.run_anonymization(parent)

    def run_anonymization(self, parent):
        new_set = parent.copy()
        try:
            anon = ModuleLoader.get_anonymization_by_name(self.config["anonymization"]["name"], self.trait)(
                self.config["anonymization"]["params"], new_set
            )
            anon.add_bg(self.sets["anonbg"])
            anon.run()
        except Exception:
            if self.config["cleanup"]:
                new_set.delete()
            raise RuntimeError("Failed to run anonymization!")
        return new_set

    def get_first_split(self, parent):
        params = {
            "original": parent.name,
            "splitter": "interid1to3",
            "params": {"rates": [self.config["rates"]["anonbg"], self.config["rates"]["attacker"]], "seed": self.config["seed"]},
            "part": 0,
        }
        bg = DatasetManager.get_matching(params)
        params["part"] = 1
        attacker = DatasetManager.get_matching(params)
        params["part"] = 2
        eval = DatasetManager.get_matching(params)

        if bg is None or attacker is None or eval is None:
            splitter = ModuleLoader.get_splitter_by_name("interid1to3")(
                {"rates": [self.config["rates"]["anonbg"], self.config["rates"]["attacker"]], "seed": self.config["seed"]}
            )
            bg, attacker, eval = splitter.run([parent])
        return bg, attacker, eval

    def get_selected_set(self, parent):
        params = {
            "original": parent.name,
            "selector": self.config["selector"]["name"],
            "params": {key: val for key, val in self.config["selector"]["params"].items() if key != "opt"} | {"seed": self.config["seed"]},
        }
        new_set = DatasetManager.get_matching(params)
        if new_set is None:
            selector = ModuleLoader.get_selector_by_name(self.config["selector"]["name"])(
                self.config["selector"]["params"] | {"seed": self.config["seed"]}
            )
            selector.set_train_set(self.sets["select_train"])
            new_set = selector.run(parent)
        else:
            self.log.info("Selection skipped. Using existing dataset " + new_set.name)
        return new_set

    def get_second_split(self, clear, anon, enroll_clear=False):
        params = {
            "original": clear.name + "|" + anon.name,
            "splitter": "intraid2to2",
            "params": {"rate": self.config["rates"]["enroll"], "seed": self.config["seed"], "enroll_clear": enroll_clear},
            "part": 0,
        }
        enroll_set = DatasetManager.get_matching(params)
        params["part"] = 1
        test_set = DatasetManager.get_matching(params)
        if enroll_set is None or test_set is None:
            splitter = ModuleLoader.get_splitter_by_name("intraid2to2")(
                {"rate": self.config["rates"]["enroll"], "seed": self.config["seed"], "enroll_clear": enroll_clear}
            )
            enroll_set, test_set = splitter.run([clear, anon])
        else:
            self.log.info("Splitting skipped. Using existing datasets " + enroll_set.name + " & " + test_set.name)
        return enroll_set, test_set
