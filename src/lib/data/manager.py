import yaml
import collections.abc
import os
import os.path
from .set import Dataset


class DatasetManager:
    @staticmethod
    def get_all():
        base = os.path.join(os.getcwd(), "data")
        sets = {}
        for set in [f for f in os.listdir(base) if f[-10:] == ".meta.yaml"]:
            with open(os.path.join(base, set), "r") as file:
                sets[set[:-10]] = yaml.load(file, Loader=yaml.SafeLoader)
        return sets

    @staticmethod
    def get_matching(config):
        sets = DatasetManager.get_all()
        for name, meta in sets.items():
            if DatasetManager.match_level(meta, config):
                set = Dataset(name)
                return set
        return None

    @staticmethod
    def match_level(meta, config):
        for k, v in config.items():
            try:
                if isinstance(meta[k], collections.abc.Mapping) and isinstance(v, collections.abc.Mapping):
                    if not DatasetManager.match_level(meta[k], v):
                        return False
                else:
                    if meta[k] != v:
                        return False
                        break
            except KeyError:
                return False
                break
        return True

    @staticmethod
    def exists(name):
        sets = DatasetManager.get_all()
        return name in sets.keys()
