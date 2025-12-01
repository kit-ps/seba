import logging
import random

from ..lib.result import ResultSet


class Inference:
    type = None
    metrics = []
    nout = 0
    random = False

    def __init__(self, config, seed, context):
        self.config = config
        self.context = context
        self.log = logging.getLogger("seba.recognition")
        self.validate_config()
        if self.random:
            random.seed(a=seed)
        self.init()

    def init(self):
        pass

    def validate_config(self):
        pass

    def train(self, set):
        pass

    def cleanup(self):
        pass

    def run(self, in_sets):
        pass


class Classification(Inference):
    type = "classification"
    metrics = ["accuracy"]

    def run(self, in_sets):
        self.enroll(in_sets[0])
        rs = self.classify(in_sets[1])
        self.cleanup()
        return rs

    def enroll(self, set):
        pass

    def classify(self, set):
        results = ResultSet.new(folder="results/")
        self.log.info("Results ID: " + results.id)

        self.log.info("Running recognition on set " + set.name)
        results = self.classify_all(set, results)
        results.save_context(self.context)
        return results

    def classify_all(self, set, results):
        for point in set[:]:
            results.append(self.classify_point(point))
        return results

    def classify_point(self, point):
        raise NotImplementedError()


class Comparison(Inference):
    type = "comparison"
    metrics = ["distance"]

    def run(self, set1, set2, save_results):
        rs = self.compare(set1, set2, save_results)
        self.cleanup()

    def compare(self, orig_set, new_set, save_results):
        results = ResultSet.new(folder="results/", save=save_results)
        self.log.info("Results ID: " + results.id)

        self.log.info("Running utility on original set {} and new set {}".format(orig_set.name, new_set.name))
        return self.compare_all(orig_set, new_set, results)

    def compare_all(self, orig_set, new_set, results):
        for point in orig_set[:]:
            results.append(self.compare_point(point, new_set[point.identity, point.session, point.pointname]))
        return results

    def compare_point(self, old_point, new_point):
        raise NotImplementedError()
