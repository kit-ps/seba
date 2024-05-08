#!/usr/bin/env python3

import logging
import os
import os.path
import sys
import yaml
import code
import copy
import time
import argparse
from src.lib.run import Run

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("logs/" + time.strftime("%Y-%m-%d_%H-%M") + ".log"), logging.StreamHandler(sys.stdout)],
    force=True,
)
_LOGGER = logging.getLogger("seba")
_CONFIG = []

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (0x = INFO, 1x = DEBUG, 2x = ALL)")
parser.add_argument("-i", "--interactive", dest="interactive", default=False, action="store_true", help="Enter interactive mode")
parser.add_argument(
    "-s",
    "--save-result",
    dest="save_result",
    default=False,
    action="store_true",
    help="Store the full result file and not just the calculated metrics.",
)
parser.add_argument("config", default=None, help="The config file to use.")
args = parser.parse_args()

if args.verbose >= 1:
    _LOGGER.setLevel(10)

if args.interactive:
    code.interact(local=dict(globals(), **locals()))
    sys.exit(0)

if not os.path.exists(args.config):
    _LOGGER.critical("Config file does not exist!")
    sys.exit(1)

try:
    with open(args.config, "r") as file:
        _CONFIG = yaml.load(file, Loader=yaml.SafeLoader)
except Exception as e:
    _LOGGER.critical("Failed to load config")
    _LOGGER.exception(e)
    sys.exit(1)

for cfg in _CONFIG:
    for round in range(cfg["repeat"]):
        try:
            run = Run(copy.deepcopy(cfg["config"]), round=round, save_result=args.save_result)
            run.run()
        except Exception as e:
            _LOGGER.warning("Run failed.")
            _LOGGER.exception(e)
            continue
