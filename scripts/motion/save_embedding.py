#!/usr/bin/env python3

"""Create a PCA embedding of the motion data and adds it to the metadata of each data point. Required for multiple
selection strategies.

Result: Update of the "embedding" in the metadata file of each data point.

Input: Path to folder with already framework compliant data set.
"""


import sys
import json
import numpy as np
from sklearn.decomposition import PCA
from src.lib.data.set import Dataset

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error, expected dataset name and model name as arguments.")
        sys.exit(1)

    setname = sys.argv[1]
    set = Dataset(setname)

    data = np.array([e.load() for e in list(set.datapoints.values())])

    data = data.reshape(data.shape[0], -1)

    pca = PCA(n_components=4)
    pca.fit(data)

    i = 0
    for mocap in set.datapoints.values():
        try:
            res = pca.transform(data[i].reshape(1, -1)).flatten().tolist()
            res = json.loads(json.dumps(res))
            mocap.save_attr("embedding", res)
        except ValueError as e:
            print(e)
        i = i + 1
