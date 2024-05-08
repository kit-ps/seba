#!/usr/bin/env python3

import argparse
import json
import logging

from src.lib.data.set import Dataset
from src.lib.module_loader import ModuleLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

_LOGGER = logging.getLogger("seba")

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="count", default=0)
parser.add_argument("-b", "--background", default=None, help="The background data set to use.")
parser.add_argument("-o", "--output", default=None, help="Name of the anonymized data set.")
parser.add_argument("-p", "--params", default="{}", help="Paramters for the anonymization method.")
parser.add_argument("anonymization", help="The name of the anonymization method to use.")
parser.add_argument("dataset", help="The data set to anonymize")
args = parser.parse_args()

if args.verbose >= 1:
    _LOGGER.setLevel(10)

set = Dataset(args.dataset)
newset = set.copy(newname=args.output)

anon = ModuleLoader.get_anonymization_by_name(args.anonymization, set.meta["trait"])(json.loads(args.params), newset)
if args.background is not None:
    bgset = Dataset(args.background)
    anon.add_bg(bgset)
anon.run()
