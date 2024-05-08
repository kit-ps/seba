import os
import os.path
import yaml
import shutil
import uuid
import logging
import time
from .point import Datapoint
from .identity import Identity


class Dataset:
    def __init__(self, name):
        self.log = logging.getLogger("seba.data")
        self.identities = {}
        self.datapoints = {}
        self.name = name
        self.log.debug("Loading dataset " + name)
        self.reload_meta()

        base = os.path.join(os.getcwd(), "data")
        self.folder = os.path.join(base, name)
        self.scan_folder()

    def reload_meta(self):
        base = os.path.join(os.getcwd(), "data")
        with open(os.path.join(base, self.name + ".meta.yaml"), "r") as file:
            self.meta = yaml.load(file, Loader=yaml.SafeLoader)

    def scan_folder(self):
        datapointclasses = Datapoint.__subclasses__()
        files = os.listdir(self.folder)
        files.sort()
        self.log.debug("Checking " + str(len(files)) + " files.")

        for file in files:
            for cls in datapointclasses:
                if cls.is_datapoint(file):
                    if not file.split(".")[0] in self.identities:
                        self.identities[file.split(".")[0]] = Identity(self.folder, file.split(".")[0])
                    identity = self.identities[file.split(".")[0]]
                    self.datapoints[".".join(file.split(".")[:2])] = cls(self.folder, self.meta, identity, file.split(".")[1])
                    identity.npoints += 1
                    continue
        self.log.info(
            "Loading dataset "
            + self.name
            + " successful. Found "
            + str(len(self.datapoints))
            + " datapoints for "
            + str(len(self.identities))
            + " identities."
        )

    def point_by_id(self):
        return list(map(lambda id: list(filter(lambda point: point.startswith(id + "."), self.datapoints.keys())), self.identities.keys()))

    def copy(self, only_points=False, only_ids=False, newname=None, softlinked=False):
        if newname is None:
            newname = str(uuid.uuid4())
        base = os.path.join(os.getcwd(), "data")
        folder = os.path.join(base, newname)
        os.mkdir(folder)

        allowed_ext = ["yaml"]
        for cls in Datapoint.__subclasses__():
            allowed_ext.append(cls.ext)

        self.log.info("Creating new dataset " + newname + ", copy of " + self.name)

        for file in os.listdir(self.folder):
            if not only_ids or file.split(".")[0] in only_ids:
                if (not only_points or ".".join(file.split(".")[:2]) in only_points or file.split(".")[1] == "yaml") and (
                    file.split(".")[-1] in allowed_ext
                ):
                    if softlinked:
                        os.symlink(os.path.join(self.folder, file), os.path.join(folder, file))
                    else:
                        shutil.copy(os.path.join(self.folder, file), os.path.join(folder, file))

        meta = {"name": newname, "original": self.name, "trait": self.meta["trait"]}
        if "original_meta" not in self.meta:
            meta["original_meta"] = self.meta
        else:
            meta["original_meta"] = self.meta["original_meta"]
        if softlinked:
            meta["softlinked"] = True
        with open(os.path.join(base, newname + ".meta.yaml"), "w") as file:
            file.write("---\n" + yaml.dump(meta))

        if softlinked:
            return SoftlinkDataset(newname)
        else:
            return Dataset(newname)

    def save_meta(self):
        base = os.path.join(os.getcwd(), "data")
        with open(os.path.join(base, self.name + ".meta.yaml"), "w") as file:
            file.write("---\n" + yaml.dump(self.meta))

    def delete(self):
        self.log.warn("Deleting data set " + self.name)
        time.sleep(10)
        shutil.rmtree(self.folder)
        base = os.path.join(os.getcwd(), "data")
        os.remove(os.path.join(base, self.name + ".meta.yaml"))


class SoftlinkDataset(Dataset):
    pass
