from .abstract import AbstractMotionAnonymization

import numpy as np


class ResamplingAnonymization(AbstractMotionAnonymization):
    """Resample the gait sequence to a specified number of frames

    Required pips:
        none
    Parameters:
        (int) fps: The target fps to which should be resampled.
        (int) total_number_of_frames: As an alternative the resulting number of total frames can be given.
    """

    name = "resampling"

    def validate_config(self):
        if "fps" not in self.config:
            self.log.info("FPS not specified, taking default value 60")
            self.config["fps"] = 60
        if "fps" not in self.dataset.meta["original_meta"]:
            self.log.error("dataset metadata does not contain an fps value")

    def anonymize(self, mocap, data):
        new_poses = []

        original_fps = self.dataset.meta["original_meta"]["fps"]

        if "total_number_of_frames" in self.config:
            target_fps = self.config["total_number_of_frames"] * (original_fps / len(data))
        else:
            target_fps = self.config["fps"]

        scaling_factor = target_fps / original_fps
        number_of_total_frames = int(len(data) * scaling_factor)

        new_indices = np.linspace(0, len(data) - 1, num=number_of_total_frames)
        for i in range(number_of_total_frames):
            start = int(new_indices[i] // 1)
            decimals = new_indices[i] % 1
            end = int(start + 1)

            if decimals == 0:
                new_poses.append(data[start])
            else:
                tmp = (data[start] * (1 - decimals)) + (data[end] * decimals)
                new_poses.append(tmp)

        return np.array(new_poses)
