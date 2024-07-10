"""
Create a mixture of two datasets
Run from project root via python -m scripts.dataset.mix <anon_dataset> <clear_dataset> <percentage> <seed>
"""

import sys
import random
import os
from src.lib.data.set import Dataset

if __name__ == "__main__":
    anon_set = Dataset(sys.argv[1])
    clear_set = Dataset(sys.argv[2])
    perc = int(sys.argv[3])

    random.seed(a=sys.argv[4])

    result_set = clear_set.copy(newname=(anon_set.name + "-p" + str(perc)), softlinked=True)

    for identity in result_set.point_by_id():
        random.shuffle(identity)
        split = int((perc / 100) * len(identity))
        for img in identity[:split]:
            os.remove(result_set.datapoints[img].get_path())
            os.symlink(anon_set.datapoints[img].get_path(), result_set.datapoints[img].get_path())
