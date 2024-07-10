#!/usr/bin/env python3
"""
Generate run config files from experiment config files.
"""

import argparse
import yaml
import copy


def generate_single_config(base):
    configs = []
    config = copy.deepcopy(base)
    config["privacy"] = {"name": "arcface", "params": {"opt": {"num_gpus": 1}}}
    configs.append(config)

    if not config["exp"] == "deanon":
        config = copy.deepcopy(base)
        config["rates"]["enroll"] = 1.0
        config["utility"] = {"name": "detection", "params": {}}
        configs.append(config)
    return configs


def generate_configs(exp_config):
    configs = []
    base = {
        "cleanup": True,
        "dataset": exp_config["dataset"],
        "rates": exp_config["rates"],
        "exp": exp_config["exp"],
        "seed": exp_config["seed"],
    }

    if "deanonymization" in exp_config:
        base["deanonymization"] = exp_config["deanonymization"]

    if "selector" in exp_config:
        base["selector"] = exp_config["selector"]

    for anonymization in exp_config["anonymizations"]:
        base["anonymization"] = anonymization
        param = None
        for k, v in base["anonymization"]["params"].items():
            if type(v) is list:
                param = k

        if param is not None:
            for p in copy.deepcopy(base["anonymization"]["params"][param]):
                base["anonymization"]["params"][param] = p
                configs.append(generate_single_config(base))
        else:
            configs.append(generate_single_config(base))

    return list(map(lambda x: {"repeat": 1, "config": x}, [x for y in configs for x in y]))


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default=None, help="The experiment configuration file to use.")
    return parser


parser = create_parser()
__doc__ += parser.format_help()

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config, "r") as file:
        exp_config = yaml.load(file, Loader=yaml.SafeLoader)

    run_config = generate_configs(exp_config)

    with open(args.config.replace(".yaml", ".runconfig.yaml"), "w") as file:
        file.write("---\n" + yaml.dump(run_config))
