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
    if "-normx" in info["name"]:
        return False, "normx"
    if "webface" in info["name"]:
        return False, "named webface"
    if 'anonymization' in info:
        if info['anonymization'] in ['ciagan', 'deepprivacy', 'dppix', 'dpsamp', 'fawkes', 'krtio', 'ksamepixel', 'ksameeigen']:
            return False, "anon:" + info['anonymization']
        else:
            return True, "anon:" + info['anonymization']
    if 'deanonymization' in info:
        if info['deanonymization'] in ['denoisenlmeans', 'dicsr', 'motiondeblurring', 'mprnet', 'normsparsity', 'rldeconv', 'stripformer', 'uwiener', 'wavelet', 'wiener2']:
            return False, "deanon:" + info['deanonymization']
        else:
            return True, "deanon:" + info['deanonymization']
    if 'splitter' in info:
        return True, "splitter"
    if 'selector' in info:
        return True, "selector"
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
