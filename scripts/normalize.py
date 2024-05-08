# Normalize the images from a dataset and save them to a new dataset
# Run from project root via python -m scripts.normalize_imgs <datasetname> [targetsize]
# Based on the pre-recognition pipeline in DeepFace (https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py#L119)

# NOTE: In rare cases, floating point arithmetic might lead to images being completely black, to fix this, apply scripts/retinaface.patch to your retinaface installation.

import sys
import json
import numpy as np
import cv2
import os
from src.lib.data.set import Dataset
from tqdm import tqdm
import logging

from retinaface import RetinaFace
from retinaface.commons import postprocess


def get_largest_face(info):
    sizes = []
    for face in info.values():
        area = face["facial_area"]
        size = (area[2] - area[0]) * (area[3] - area[1])
        sizes.append(size)
    largest = int(np.argmax(sizes))
    return list(info.values())[largest]


def normalize_image(point, model, interactive=False):
    img = cv2.imread(point.get_path())

    info = RetinaFace.detect_faces(img, model=model)

    if not type(info) == dict:
        if point.bbox == None or point.landmarks == None:
            raise AttributeError("no face found and no bbox or landmark meta in " + point.get_path())
        else:
            info = {
                "face_1": {
                    "facial_area": [point.bbox["left"], point.bbox["top"], point.bbox["right"], point.bbox["bottom"]],
                    "landmarks": point.landmarks,
                }
            }

    if not interactive:
        face = get_largest_face(info)
    else:
        print(info)
        x = input("Face # -->")
        face = info["face_" + x]
    area = face["facial_area"]  # (left, top, right, bottom)

    add_top = add_bottom = add_left = add_right = 0

    if (area[2] - area[0]) > (area[3] - area[1]):
        # more wide
        diff = (area[2] - area[0]) - (area[3] - area[1])
        add_top = diff // 2
        add_bottom = diff - add_top
    else:
        # more tall
        diff = (area[3] - area[1]) - (area[2] - area[0])
        add_left = diff // 2
        add_right = diff - add_left

    new_area = [area[0] - add_left, area[1] - add_top, area[2] + add_right, area[3] + add_bottom]

    border = int((new_area[2] - new_area[0]) * 0.25)
    img = cv2.copyMakeBorder(img, 2 * border, 2 * border, 2 * border, 2 * border, cv2.BORDER_REPLICATE)
    img = img[new_area[1] + border : new_area[3] + 3 * border, new_area[0] + border : new_area[2] + 3 * border]

    landmarks = face["landmarks"]
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    nose = landmarks["nose"]

    # newer versions of retinaface also require the nose position for alignment
    # the following allows all versions to work
    try:
        img = postprocess.alignment_procedure(img, right_eye, left_eye, nose)
    except TypeError:
        img = postprocess.alignment_procedure(img, right_eye, left_eye)

    img = img[border:-border, border:-border]

    img = cv2.resize(img, target_size)

    cv2.imwrite(point.get_path()[:-4] + ".png", img)
    point.save_attr("bbox", {"top": 0, "left": 0, "bottom": target_size[0] - 1, "right": target_size[0] - 1})
    if not point.get_path() == point.get_path()[:-4] + ".png":
        os.remove(point.get_path())


if len(sys.argv) < 2:
    logging.error("Error, expected dataset name as argument.")
    sys.exit(1)
setname = sys.argv[1]
set = Dataset(setname)
model = RetinaFace.build_model()

target_size = (224, 224)
if len(sys.argv) > 2:
    target_size = (int(sys.argv[2]), int(sys.argv[2]))

interactive = False
if len(sys.argv) > 3:
    interactive = True

if not interactive:
    newset = set.copy(newname=set.name + "-normx")
    for point in tqdm(newset.datapoints.values()):
        try:
            normalize_image(point, model)
        except BaseException as e:
            logging.error("\n\n")
            logging.error("ERROR: failed to normalize image " + point.get_path())
            logging.exception(e)
else:
    point = set.datapoints[sys.argv[3]]
    normalize_image(point, model, True)
