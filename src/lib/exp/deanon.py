from ..data.manager import DatasetManager
from ..module_loader import ModuleLoader
from .abstractanon import AbstractAnonExperiment


class DeanonExperiment(AbstractAnonExperiment):
    def run(self):
        self.sets["anonbg"], self.sets["attacker"], self.sets["eval"] = self.get_first_split(self.sets["orig"])
        self.sets["train"] = self.sets["attacker"]
        self.sets["anon_attacker"] = self.get_anonymized_set(self.sets["attacker"])
        self.sets["select_train"] = self.sets["anon_attacker"]
        self.sets["anon"] = self.get_anonymized_set(self.sets["eval"])
        self.sets["select"] = self.get_selected_set(self.sets["anon"])
        self.sets["deanon"] = self.get_deanonymized_set(self.sets["select"])

        self.sets["enroll"], self.sets["test"] = self.get_second_split(self.sets["eval"], self.sets["deanon"], enroll_clear=True)

        self.run_evaluation()
        self.run_metrics()
        self.save_metrics()

    def get_deanonymized_set(self, parent):
        config = self.config["deanonymization"]
        params = {
            "original": parent.name,
            "deanonymization": config["name"],
            "params": {key: val for key, val in config["params"].items() if key != "opt"},
        }
        new_set = DatasetManager.get_matching(params)
        if new_set is None:
            new_set = parent.copy()
            try:
                deanonymization = ModuleLoader.get_deanonymization_by_name(config["name"], self.trait)(config["params"])
                deanonymization.train(self.sets["attacker"], self.sets["anon_attacker"])
                deanonymization.run(new_set)
                deanonymization.cleanup()
                del deanonymization
            except Exception:
                if self.config["cleanup"]:
                    new_set.delete()
                raise RuntimeError("Failed to run de-anonymization!")
        else:
            self.log.info("Deanonymization skipped. Using existing dataset " + new_set.name)
        return new_set
