# Run a variety of sanity checks on a face image dataset
# Run from project root via python -m scripts.face.check_dataset <datasetname>

import sys
import cv2
from src.lib.data.set import Dataset

SET_CHECKS = []
IMAGE_CHECKS = ["check_image_bbox", "check_image_onecolored"]


def check_image_bbox(set, point, image):
    if point.bbox["top"] < 0:
        return "top bbox < 0"
    if point.bbox["left"] < 0:
        return "left bbox < 0"
    if point.bbox["bottom"] - point.bbox["top"] <= 0:
        return "vertical bbox incorrect"
    if point.bbox["right"] - point.bbox["left"] <= 0:
        return "horizontal bbox incorrect"
    if point.bbox["bottom"] >= image.shape[0]:
        return "bottom bbox >= image shape"
    if point.bbox["right"] >= image.shape[1]:
        return "right bbox >= image shape"


def check_image_onecolored(set, point, image):
    if image.sum() == 0:
        return "image is completely black"
    if image.std(axis=(0, 1)).sum() == 0:
        return "image is completely one-colored."
    if image[0].std(axis=0).sum() == 0:
        return "first line one-colored."
    if image[image.shape[0] - 1].std(axis=0).sum() == 0:
        return "last line one-colored."
    if image[:, 0].std(axis=0).sum() == 0:
        return "first column one-colored."
    if image[:, image.shape[1] - 1].std(axis=0).sum() == 0:
        return "last column one-colored."


def image_checks(set):
    for point in set.datapoints.values():
        image = cv2.imread(point.get_path())
        for func in IMAGE_CHECKS:
            r = globals()[func](set, point, image)
            if r is not None:
                print("FAIL: Image {}, {}".format(point.get_path(), r))


def set_checks(set):
    for func in SET_CHECKS:
        r = globals()[func](set)
        if r is not None:
            print("FAIL: {}".format(r))


setname = sys.argv[1]
set = Dataset(setname)
set_checks(set)
image_checks(set)
