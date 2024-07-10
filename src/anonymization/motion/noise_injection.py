from .abstract import AbstractMotionAnonymization

import numpy as np


class Noise_injectionAnonymization(AbstractMotionAnonymization):
    """Apply noise injection

    Required pips:
        none
    Parameters:
        - (string) distribution: one of ["normal", "laplace", "uniform"]
        - (float) scaling: a scalar to scale the noise added
        - (string) type: one of ["direct", "indirect"]
    """

    name = "noise_injection"

    def validate_config(self):
        if "distribution" not in self.config or self.config["distribution"] not in ["normal", "laplace", "uniform"]:
            self.log.info("Missing distribution value, taking default 'uniform'")
            self.config["distribution"] = "uniform"
        if "scaling" not in self.config:
            self.log.info("Missing scaling value, taking default 1")
            self.config["scaling"] = 1
        if "type" not in self.config:
            self.log.info("Missing type value, taking default direct")
            self.config["type"] = "direct"

    def anonymize(self, mocap, data):
        size = len(data) * len(data[0])

        rand = np.random.default_rng().uniform(0, 1, size)
        if self.config["distribution"] == "laplace":
            rand = np.random.default_rng().laplace(0, 1, size)
        if self.config["distribution"] == "normal":
            rand = np.random.default_rng().normal(0, 1, size)

        new_data = []
        if self.config["type"] == "direct":
            std_dev = data.std(0)

            for i in range(len(data)):
                new_data.append(rand[i * len(data[0]) : (i + 1) * len(data[0])] * std_dev * self.config["scaling"] + data[i])

        if self.config["type"] == "indirect":
            tmp_diff = []
            for i in range(1, len(data)):
                tmp_diff.append(data[i] - data[i - 1])

            tmp_diff = np.array(tmp_diff)
            new_data.append(data[0] + tmp_diff[0])
            for i in range(1, len(tmp_diff)):
                new_data.append(new_data[i - 1] + tmp_diff[i])

        new_data = np.array(new_data)
        return new_data
