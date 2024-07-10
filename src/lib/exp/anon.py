from .abstractanon import AbstractAnonExperiment


class AnonExperiment(AbstractAnonExperiment):
    def run(self):
        self.sets["anonbg"], self.sets["attacker"], self.sets["eval"] = self.get_first_split(self.sets["orig"])
        self.sets["train"] = self.get_anonymized_set(self.sets["attacker"])
        self.sets["select_train"] = self.sets["train"]
        self.sets["anon"] = self.get_anonymized_set(self.sets["eval"])
        self.sets["select"] = self.get_selected_set(self.sets["anon"])

        self.sets["enroll"], self.sets["test"] = self.get_second_split(self.sets["eval"], self.sets["select"])

        self.run_evaluation()
        self.run_metrics()
        self.save_metrics()
