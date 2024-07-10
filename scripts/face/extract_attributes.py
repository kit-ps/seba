"""
Run DeepFace attribute extraction on a dataset and save the extracted attributes to the image metadata files.
Run from project root via python -m scripts.face.extract_attributes <datasetname>
"""

import sys
import json
from deepface import DeepFace
from src.lib.data.set import Dataset

from tqdm import tqdm

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error, expected dataset name as argument.")
        sys.exit(1)

    setname = sys.argv[1]
    set = Dataset(setname)

    i = 0
    n = len(set.datapoints)

    for img in tqdm(set.datapoints.values()):
        i += 1
        print("Image {}/{}.".format(i, n))
        try:
            res = DeepFace.analyze(
                img_path=img.get_path(),
                actions=["age", "race", "gender"],
                detector_backend="retinaface",
                already_normalized=True,
            )
            res = json.loads(json.dumps(res))
            img.save_attr("deepface", res)
        except ValueError as e:
            print(e)
