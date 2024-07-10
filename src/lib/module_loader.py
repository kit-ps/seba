from ..anonymization.abstract import AbstractAnonymization
from ..deanonymization.abstract import AbstractDeanonymization
from ..selector.abstract import AbstractSelector
from ..splitter.abstract import AbstractSplitter
from ..privacy.abstract import AbstractPrivacy
from ..utility.abstract import AbstractUtility
from ..metric.abstract import AbstractMetric
from ..lib.exp.abstract import AbstractExperiment
from ..lib.inference import Inference

import importlib


class ModuleLoader:
    @staticmethod
    def get_splitter_by_name(name):
        class_name = name.capitalize() + "Splitter"
        sl_module = importlib.import_module("src.splitter." + name)
        sl_class = getattr(sl_module, class_name)
        if not issubclass(sl_class, AbstractSplitter):
            raise AttributeError("Splitter does not inherit AbstractSplitter.")
        return sl_class

    @staticmethod
    def get_selector_by_name(name):
        class_name = name.capitalize() + "Selector"
        sl_module = importlib.import_module("src.selector." + name)
        sl_class = getattr(sl_module, class_name)
        if not issubclass(sl_class, AbstractSelector):
            raise AttributeError("Selector does not inherit AbstractSelector.")
        return sl_class

    @staticmethod
    def get_classification_by_name(name, trait):
        class_name = name.capitalize() + "Classification"
        r_module = importlib.import_module("src.privacy." + trait + "." + name)
        r_class = getattr(r_module, class_name)
        if not issubclass(r_class, AbstractPrivacy):
            raise AttributeError("Recognition does not inherit AbstractPrivacy.")
        if not issubclass(r_class, Inference):
            raise AttributeError("Recognition does not inherit Inference.")
        return r_class

    @staticmethod
    def get_utility_by_name(name, trait):
        class_name = name.capitalize() + "Utility"
        u_module = importlib.import_module("src.utility." + trait + "." + name)
        u_class = getattr(u_module, class_name)
        if not issubclass(u_class, AbstractUtility):
            raise AttributeError("Utility does not inherit AbstractRecognition.")
        if not issubclass(u_class, AbstractUtility):
            raise AttributeError("Utility does not inherit Inference.")
        return u_class

    @staticmethod
    def get_anonymization_by_name(name, trait):
        class_name = name.capitalize() + "Anonymization"
        anon_module = importlib.import_module("src.anonymization." + trait + "." + name)
        anon_class = getattr(anon_module, class_name)
        if not issubclass(anon_class, AbstractAnonymization):
            raise AttributeError("Anonymization does not inherit AbstractAnonymization.")
        return anon_class

    @staticmethod
    def get_deanonymization_by_name(name, trait):
        class_name = name.capitalize() + "Deanonymization"
        deanon_module = importlib.import_module("src.deanonymization." + trait + "." + name)
        deanon_class = getattr(deanon_module, class_name)
        if not issubclass(deanon_class, AbstractDeanonymization):
            raise AttributeError("Deanonymization does not inherit AbstractDeanonymization.")
        return deanon_class

    @staticmethod
    def get_metric_by_name(name):
        class_name = name.capitalize() + "Metric"
        metric_module = importlib.import_module("src.metric." + name)
        metric_class = getattr(metric_module, class_name)
        if not issubclass(metric_class, AbstractMetric):
            raise AttributeError("Metric does not inherit AbstractMetric.")
        return metric_class

    @staticmethod
    def get_exp_by_name(name):
        class_name = name.capitalize() + "Experiment"
        exp_module = importlib.import_module("src.lib.exp." + name)
        exp_class = getattr(exp_module, class_name)
        if not issubclass(exp_class, AbstractExperiment):
            raise AttributeError("Experiment does not inherit AbstractExperiment.")
        return exp_class
