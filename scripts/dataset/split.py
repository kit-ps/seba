# Split a dataset into similarly sized parts
# Run from project root via python -m scripts.dataset.split <dataset> <no_of_parts>

import sys
import math
from src.lib.data.set import Dataset


def chunks(lst, n):
    r = []
    chunksize = math.ceil(len(lst) / n)
    i = 0
    while (i * chunksize) < len(lst):
        r.append(lst[i * chunksize : (i + 1) * chunksize])
        i += 1
    return r


set = Dataset(sys.argv[1])
points = chunks(list(set.datapoints.keys()), int(sys.argv[2]))
for i in range(len(points)):
    set.copy(only_points=points[i], newname=set.name + "-pt" + str(i))
