#!/bin/env python3

'''
cleanup.py
'''

import os
import yaml
import shutil


def delete_set(info):
    if info["original"] == True:
        return False, "original"
    if 'type' in info:
        return False, 'type[{}]'.format(info['type'])
    return True, "unknown"

if __name__ == "__main__":
    base = os.path.join(os.getcwd(), "data")

    for set in [f for f in os.listdir(base) if f[-10:] == ".meta.yaml"]:
        with open(os.path.join(base, set), "r") as file:
            info = yaml.load(file, Loader=yaml.SafeLoader)
        delete, info = delete_set(info)

        if delete:
            print("Dataset: " + set[:-10] + " TYPE[" + info + "] - DELETING")
            shutil.rmtree(os.path.join(base, set[:-10]))
            os.remove(os.path.join(base, set))
        else:
            print("Dataset: " + set[:-10] + " TYPE[" + info + "] - KEEPING")
