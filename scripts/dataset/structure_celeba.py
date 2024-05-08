#!/usr/bin/env python3

"""Create dir structure for CelebA dataset

Original: One Folder with all pictures + identity.txt + attributes.txt + bbox.txt
Result: Files named idname.pointname.jpg + metadata.yaml per picture.

Input: Path to folder with celeba data.
"""

import os.path
import sys
import shutil
import yaml


if len(sys.argv) < 2 or not sys.argv[1]:
    exit(1)

folder = sys.argv[1]
if folder[-1] != "/":
    folder += "/"

if not os.path.exists(folder) or not os.path.exists(folder + "identity.txt") or not os.path.exists(folder + "attributes.txt"):
    exit(1)

identities = {}
with open(folder + "identity.txt", "r") as file:
    for line in file.read().splitlines():
        key, value = line.split()
        identities[key] = value

attributes = {}
with open(folder + "attributes.txt", "r") as file:
    lines = file.read().splitlines()
    attributes["keys"] = list(map(lambda x: x.lower(), lines[1].split()))
    for line in lines[2:]:
        values = line.split()
        attributes[values[0]] = list(map(lambda x: True if x == "1" else False, values[1:]))

with open(folder + "bbox.txt", "r") as file:
    lines = file.read().splitlines()
    for line in lines[2:]:
        values = line.split()
        attributes[values[0]].append(
            {
                "left": int(values[1]),
                "top": int(values[2]),
                "right": int(values[1]) + int(values[3]),
                "bottom": int(values[2]) + int(values[4]),
            }
        )
attributes["keys"].append("bbox")

with open(folder + "landmarks.txt", "r") as file:
    lines = file.read().splitlines()
    for line in lines[2:]:
        values = line.split()
        data = list(map(lambda x: int(x), values[1:]))
        attributes[values[0]].append(
            {"left_eye": data[0:2], "right_eye": data[2:4], "nose": data[4:6], "left_mouth": data[6:8], "right_mouth": data[8:10]}
        )
attributes["keys"].append("landmarks")

attr = {}
for pic in identities:
    shutil.move(folder + pic, folder + identities[pic] + "." + pic)
    for i in range(len(attributes["keys"])):
        attr[attributes["keys"][i]] = attributes[pic][i]
    with open(folder + identities[pic] + "." + pic[:-4] + ".yaml", "w") as file:
        file.write("---\n" + yaml.dump(attr))
