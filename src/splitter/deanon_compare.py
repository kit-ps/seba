from .deanon import DeanonSplitter


class Deanon_compareSplitter(DeanonSplitter):
    name = "deanon_compare"

    def validate_config(self):
        if "train_rate" not in self.config:
            raise AttributeError("Splitter: config: Missing train_rate")
        else:
            self.config["train_rate"] = float(self.config["train_rate"])
            if self.config["train_rate"] < 0 or self.config["train_rate"] >= 1:
                raise AttributeError("Splitter: config: train_rate not in [0,1]")

    def create_test(self, orig_set, anon_set, test_ids):
        return [orig_set.copy(only_ids=test_ids, softlinked=True), anon_set.copy(only_ids=test_ids, softlinked=True)]
