"""
Create a new original dataset by running a selector on another dataset
Run from project root via python -m scripts.dataset.select [--opts OPTS_JSON] <selector> <dataset_tocopy> <dataset_tocreate>
"""

import json
import argparse
import os
from src.lib.data.set import Dataset
from src.lib.module_loader import ModuleLoader


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opts", help="Json encoded options for the selector", default="{}")
    parser.add_argument("selector", default=None, help="Selector to use")
    parser.add_argument("dataset_tocopy", default=None, help="The name of the existing dataset to copy.")
    parser.add_argument("dataset_tocreate", default=None, help="The name of the new dataset.")
    return parser


parser = create_parser()
__doc__ += parser.format_help()

if __name__ == "__main__":
    args = parser.parse_args()

    set = Dataset(args.dataset_tocopy)
    selector = ModuleLoader.get_selector_by_name(args.selector)(json.loads(args.opts))
    new_set = selector.run(set)

    name = new_set.name
    new_set.meta = {"name": args.dataset_tocreate, "original": True, "trait": set.meta["trait"]}
    new_set.save_meta()

    base = os.path.join(os.getcwd(), "data")
    os.rename(os.path.join(base, name + ".meta.yaml"), os.path.join(base, args.dataset_tocreate + ".meta.yaml"))
    os.rename(os.path.join(base, name), os.path.join(base, args.dataset_tocreate))
