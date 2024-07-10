"""
This script copies all models that are saved in subfolders of data/ into a models/ folder and adds metadata.
This can be used to preserve models before using data/cleanup.py to remove datasets.
"""

from src.lib.data.manager import DatasetManager
from src.lib.result import ResultsManager

import shutil
import os
import yaml

if __name__ == "__main__":
    sets = DatasetManager.get_all()

    for name, meta in sets.items():
        if "models" in meta:
            for model in meta["models"]:
                try:
                    newpath = model["path"].replace("/data/", "/models/")[::-1].replace("/", "-", 1)[::-1]
                    shutil.copyfile(model["path"], newpath)
                    print("Copied " + model["path"])
                    model["path"] = newpath
                except Exception:
                    print("Failed to copy model " + model["path"] + " from dataset " + name)

            rs = ResultsManager.get_matching_by_used_dataset(meta["name"])
            if rs:
                config = ResultsManager.get_config(sorted(rs)[-1])
                config["models"] = meta["models"]
            else:
                print("WARN: no run config found. Including data set meta.")
                config = meta
            path = os.path.join("/", *newpath.split("/")[:-1], meta["name"] + ".config.yaml")
            with open(path, "w+") as f:
                f.write("---\n" + yaml.dump(config))
            print("Wrote " + str(path) + "\n\n")
