import os
import os.path
import yaml
import json
import numpy as np


class Datapoint:
    ext = "yaml"

    def __init__(self, setpath, setmetadata, identity, pointname):
        self.setpath = setpath
        self.setmetadata = setmetadata
        self.idname = identity.name
        self.identity = identity
        self.pointname = pointname
        self.attr = False

    def __getattr__(self, atr):
        self.load_attr()
        if atr not in self.attr:
            return None
        else:
            return self.attr[atr]

    def load_attr(self):
        if self.attr is False:
            # lazy loading
            if os.path.exists(os.path.join(self.setpath, self.idname + "." + self.pointname + ".yaml")):
                with open(os.path.join(self.setpath, self.idname + "." + self.pointname + ".yaml"), "r") as file:
                    self.attr = yaml.load(file, Loader=yaml.SafeLoader)
            else:
                self.attr = {}

    def save_attr(self, key, value):
        self.load_attr()
        self.attr[key] = value
        self.save_attr_file()

    def save_attr_batch(self, attributes):
        self.load_attr()
        for k, v in attributes.items():
            self.attr[k] = v
        self.save_attr_file()

    def save_attr_file(self):
        with open(os.path.join(self.setpath, self.idname + "." + self.pointname + ".yaml"), "w") as file:
            file.write("---\n" + yaml.dump(self.attr))

    def get_path(self):
        return os.path.join(self.setpath, self.get_filename())

    def get_filename(self):
        return self.idname + "." + self.pointname + "." + self.ext

    @staticmethod
    def is_datapoint(filename):
        return False


class Image(Datapoint):
    ext = "jpg"

    @staticmethod
    def is_datapoint(filename):
        return filename.split(".")[-1] == "jpg"


class PNGImage(Datapoint):
    ext = "png"

    @staticmethod
    def is_datapoint(filename):
        return filename.split(".")[-1] == "png"


class MOCAP(Datapoint):
    ext = "mocap"

    @staticmethod
    def is_datapoint(filename):
        return filename.split(".")[-1] == "mocap"

    def load_position(self):
        with open(self.get_path(), "r") as f:
            data = np.array(json.load(f))

            if "original_meta" in self.setmetadata:
                meta = self.setmetadata["original_meta"]
            else:
                meta = self.setmetadata

            range = meta["position_range"]
            data = data[:,range[0]:range[1]]

        return data

    def load_rotation(self):
        with open(self.get_path(), "r") as f:
            data = np.array(json.load(f))

            if "original_meta" in self.setmetadata:
                meta = self.setmetadata["original_meta"]
            else:
                meta = self.setmetadata

            range = meta["rotation_range"]
            data = data[:,range[0]:range[1]]

        return data

    def load(self):
        with open(self.get_path(), "r") as f:
            data = np.array(json.load(f))
        return data

    def save(self, data):
        with open(self.get_path(), "w") as f:
            json.dump(data.tolist(), f)


class Video(Datapoint):
    pass
    # other biometrics GAIT
