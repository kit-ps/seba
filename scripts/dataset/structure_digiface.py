"""
Create dir structure for DigiFace-1M dataset

Original: One Folder with subfolders for each identity which contains the images for that identity
Result: Files named idname.pointname.jpg + metadata.yaml per picture.

Input: Path to folder with digiface data.
"""

import os
import os.path
import sys
import shutil

if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1]:
        exit(1)

    folder = sys.argv[1]

    if not os.path.exists(folder):
        exit(1)

    for identity in os.listdir(folder):
        for pic in os.listdir(os.path.join(folder, identity)):
            shutil.move(os.path.join(folder, identity, pic), os.path.join(folder, identity + "." + pic))
        shutil.rmtree(os.path.join(folder, identity))
