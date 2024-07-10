"""
Merge two or more datasets
Resulting dataset will have the name of the first dataset + "-merged". Rename using scripts.dataset.rename
Run from project root via python -m scripts.dataset.merge [datasets]
"""

import sys
import os
import shutil
from src.lib.data.set import Dataset

if __name__ == "__main__":
    sets = sys.argv[1:]
    set0 = Dataset(sets[0])
    newname = set0.name + "-merged"
    if set0.name[-4:-1] == "-pt":
        newname = set0.name[:-4] + "-merged"
    set0.copy(newname=newname)

    base = os.path.join(os.getcwd(), "data")
    for set in sets[1:]:
        shutil.copytree(os.path.join(base, set), os.path.join(base, newname), dirs_exist_ok=True)
