"""
Rename a data set
Run from project root via python -m scripts.dataset.rename <current_name> <future_name>
"""

import sys
import os
from src.lib.data.set import Dataset

if __name__ == "__main__":
    base = os.path.join(os.getcwd(), "data")

    if os.path.exists(os.path.join(base, sys.argv[2] + ".meta.yaml")):
        raise AttributeError("destination exists!")

    set = Dataset(sys.argv[1])
    set.meta["name"] = sys.argv[2]
    set.save_meta()

    os.rename(os.path.join(base, sys.argv[1] + ".meta.yaml"), os.path.join(base, sys.argv[2] + ".meta.yaml"))
    os.rename(os.path.join(base, sys.argv[1]), os.path.join(base, sys.argv[2]))
