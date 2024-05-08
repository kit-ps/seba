import os
import json
import time
import uuid
import yaml


class Result:
    def __init__(self, identity, picture_id):
        self.identity = identity
        self.picture_id = picture_id
        self.recognized = []

    def add_recognized(self, recognized_id, dist=0.0):
        self.recognized.append({"id": recognized_id, "dist": float(dist)})

    def get_best_recognized(self):
        if not len(self.recognized):
            return None
        self.recognized.sort(key=lambda x: x["dist"])
        return self.recognized[0]

    def is_success(self):
        if not len(self.recognized):
            return False
        return self.get_best_recognized()["id"] == self.identity

    def is_topn_success(self, n):
        if not len(self.recognized):
            return False
        self.recognized.sort(key=lambda x: x["dist"])
        topn = list(map(lambda x: x["id"], self.recognized))[:n]
        return self.identity in topn

    def to_squashed(self, strat=None):
        if strat is None:
            return self
        r = Result(self.identity, self.picture_id)
        recog = {}
        for recognized in self.recognized:
            if not recognized["id"] in recog:
                recog[recognized["id"]] = []
            recog[recognized["id"]].append(recognized["dist"])
        for key in recog:
            if strat == "min":
                dist = min(recog[key])
            elif strat == "mean":
                dist = sum(recog[key]) / len(recog[key])
            else:
                raise AttributeError("unknown strat {}".format(strat))
            r.add_recognized(key, dist)
        return r

    def __str__(self):
        s = "Result for {} (correct: {})".format(self.picture_id, self.identity)
        self.recognized.sort(key=lambda x: x["dist"])
        for r in self.recognized:
            s += "{} ({}); ".format(r["id"], r["dist"])
        return s


class ResultSet:
    def __init__(self, filename, save=True):
        self.id = filename.split("/")[-1][:-12]
        self.filename = filename
        self.save = save
        self.results = []
        self.config = {}
        self.datasets = {}
        if os.path.isfile(self.filename):
            self.load()

    @classmethod
    def new(cls, folder="results/", save=True):
        id = time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(uuid.uuid4())[:8]
        filename = folder + id + ".results.txt"
        return ResultSet(filename=filename, save=save)

    def load(self):
        with open(self.filename, "r") as file:
            lines = file.read().splitlines()
        for line in lines:
            parts = line.split(",")
            if parts[0] == "config":
                self.config = json.loads(",".join(parts[1:]))
                continue
            if parts[0] == "dataset":
                if ":" in parts[1]:
                    self.datasets = dict(map(lambda x: x.split(":"), parts[1:]))
                else:
                    self.datasets = dict(zip(range(len(parts[1:])), parts[1:]))
                continue
            if len(parts) < 2:
                raise AttributeError("invalid syntax in results file")
            result = Result(parts[0], parts[1])
            for i in range(2, len(parts)):
                if parts[i] == "":
                    continue
                sub = parts[i].split("|")
                result.add_recognized(*sub)
            self.results.append(result)

    def save_context(self, config, datasets):
        self.config = config
        self.datasets = datasets
        if self.save:
            with open(self.filename, "a") as f:
                f.write("config," + json.dumps(config) + "\n")
                f.write("dataset," + ",".join(map(lambda x: x[0] + ":" + x[1], datasets.items())) + "\n")

    def save(self):
        for r in self.results:
            self.append_save(r)

    def append(self, result):
        self.results.append(result)
        if self.save:
            self.append_save(result)

    def append_save(self, result):
        line = result.identity + "," + result.picture_id + ","
        line += ",".join(map(lambda x: "|".join(map(lambda y: str(y), x.values())), result.recognized))
        line += "\n"
        with open(self.filename, "a") as f:
            f.write(line)

    def to_squashed(self, strat=None):
        r = ResultSet(self.filename + "-squashed")
        r.results = list(map(lambda x: x.to_squashed(strat=strat), self.results))
        return r


class ResultsManager:
    def sort_dict(item, reverse=False):
        return {
            k: ResultsManager.sort_dict(v) if isinstance(v, dict) else v
            for k, v in sorted(item.items(), key=lambda x: x[0], reverse=reverse)
        }

    def get_matching_by_used_dataset(needle):
        all = ResultsManager.sort_dict(ResultsManager.get_all(), reverse=True)
        for name, result in all.items():
            if "datasets" in result and needle in result["datasets"].values():
                return name

    def get_matching_by_config(config):
        all = ResultsManager.sort_dict(ResultsManager.get_all(), reverse=True)
        for name, result in all.items():
            if json.dumps(result["config"]) == json.dumps(config):
                return name

    def get_set_of_matching_by_configs(configs):
        all = ResultsManager.sort_dict(ResultsManager.get_all(), reverse=True)
        names = []
        for config in configs:
            config = ResultsManager.sort_dict(config)
            matched = False
            for name, result in all.items():
                if json.dumps(result["config"]) == json.dumps(config):
                    names.append(name)
                    matched = True
                    break
            if not matched:
                names.append(None)
        return names

    def get_all():
        with open("results.yaml", "r") as file:
            return yaml.load(file, Loader=yaml.SafeLoader)

    def get_config_by_name(name):
        all = ResultsManager.get_all()
        return all[name]
