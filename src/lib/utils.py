import logging
import subprocess
from contextlib import contextmanager
import sys
import os


def recursive_replace(data, needle, replace):
    if type(data) is dict:
        for key, value in data.items():
            data[key] = recursive_replace(data[key], needle, replace)
    elif type(data) is list:
        for key in range(len(data)):
            data[key] = recursive_replace(data[key], needle, replace)
    elif type(data) is str:
        data = data.replace(needle, str(replace))
    return data


def exec_ext_cmd(cmd, cwd=None, ignore=False):
    logging.getLogger("seba.cmd").info("Running an external command: " + " ".join(cmd))
    output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
    logging.getLogger("seba.cmd").debug(output.stdout)
    if output.returncode > 0 and not ignore:
        raise RuntimeError("External command returned non-zero exit-code!")


# surpress cmdline output
# from http://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
