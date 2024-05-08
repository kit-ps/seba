#!/usr/bin/env python3

import logging
import os.path
import sys
import argparse
from src.lib.result import ResultSet
from src.lib.module_loader import ModuleLoader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

logger = logging.getLogger("seba")

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="count", default=0)
parser.add_argument("results", default=None, help="The result file to use.")
parser.add_argument("metric", default="export", help="The metric to calculate.")
args = parser.parse_args()

if args.verbose >= 1:
    logger.setLevel(10)

if not os.path.exists(args.results):
    logger.critical("Results file does not exist!")
    sys.exit(1)

try:
    resultset = ResultSet(args.results, save=False)
    metric = ModuleLoader.get_metric_by_name(args.metric)(resultset)
    metric.run()
except Exception as e:
    logger.critical("Failed to redo metric")
    logger.exception(e)
    sys.exit(1)
