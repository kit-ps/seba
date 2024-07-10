from .abstract import AbstractMotionAnonymization

import numpy as np


class CoarseningAnonymization(AbstractMotionAnonymization):
    """Apply coarsening

    Required pips:
        numpy
    Parameters:
        - (int) coarsening_factor: Gives the factor for the coarsening, 10 would only keep every 10th point.
        - (string) coarsening_type: one of "time" or "precision"
        - (int) coarsening_decimals: How many decimals to keep, can be negative.
    """

    name = "coarsening"

    def validate_config(self):
        if "coarsening_type" not in self.config:
            self.log.info("Missing coarsening type, taking default time")
            self.config["coarsening_type"] = "time"

        if "coarsening_factor" not in self.config and self.config["coarsening_type"] == "time":
            self.log.info("Missing coarsening_factor value, taking default 10")
            self.config["coarsening_factor"] = 10

        if "coarsening_decimals" not in self.config and self.config["coarsening_type"] == "precision":
            self.log.info("Missing coarsening_decimals value, taking default 0")
            self.config["coarsening_decimals"] = 0

    def anonymize(self, mocap, data):
        if self.config["coarsening_type"] == "precision":
            data = np.around(data, decimals=self.config["coarsening_decimals"])

        if self.config["coarsening_type"] == "precision_inverted":
            data = data % self.config["coarsening_decimals"]

        if self.config["coarsening_type"] == "time":
            for i in range(len(data) // self.config["coarsening_factor"] - 1):
                diff = data[(i + 1) * self.config["coarsening_factor"]] - data[i * self.config["coarsening_factor"]]

                for ii in range(self.config["coarsening_factor"]):
                    data[i * self.config["coarsening_factor"] + ii] = data[i * self.config["coarsening_factor"]] + diff * (
                        ii / self.config["coarsening_factor"]
                    )

        return data
