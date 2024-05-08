import logging

from ..lib.result import ResultSet


class Inference:
    type = None
    metrics = []

    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger("seba.recognition")
        self.validate_config()
        self.init()

    def init(self):
        pass

    def validate_config(self):
        pass

    def train(self, set):
        pass

    def cleanup(self):
        pass

    def run(self, set1, set2, save_results):
        pass


class Classification(Inference):
    type = "classification"
    metrics = ["accuracy"]

    def run(self, set1, set2, save_results):
        self.enroll(set1)
        return self.classify(set2, save_results)

    def classify(self, set, save_results):
        results = ResultSet.new(folder="results/", save=save_results)
        self.log.info("Results ID: " + results.id)

        self.log.info("Running recognition on set " + set.name)
        return self.classify_all(set, results)

    def classify_all(self, set, results):
        for name, point in set.datapoints.items():
            results.append(self.classify_point(point))
        return results

    def classify_point(self, point):
        raise NotImplementedError()


class Comparison(Inference):
    type = "comparison"
    metrics = ["distance"]

    def run(self, set1, set2, save_results):
        return self.compare(set1, set2, save_results)

    def compare(self, orig_set, new_set, save_results):
        results = ResultSet.new(folder="results/", save=save_results)
        self.log.info("Results ID: " + results.id)

        self.log.info("Running utility on original set {} and new set {}".format(orig_set.name, new_set.name))
        return self.compare_all(orig_set, new_set, results)

    def compare_all(self, orig_set, new_set, results):
        for key in orig_set.datapoints.keys():
            results.append(self.compare_point(orig_set.datapoints[key], new_set.datapoints[key]))
        return results

    def compare_point(self, old_point, new_point):
        raise NotImplementedError()
