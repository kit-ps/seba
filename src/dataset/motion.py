from .individualfile import FileDataPoint, IndividualfileDataset

import numpy as np
import json

class MotionDataPoint(FileDataPoint):
    def load_position(self):
        with open(self.get_path(), "r") as f:
            data = np.array(json.load(f))

            if "original_meta" in self.setmetadata:
                meta = self.setmetadata["original_meta"]
            else:
                meta = self.setmetadata

            range = meta["position_range"]
            data = data[:, range[0] : range[1]]

        return data

    def load_rotation(self):
        with open(self.get_path(), "r") as f:
            data = np.array(json.load(f))

            if "original_meta" in self.setmetadata:
                meta = self.setmetadata["original_meta"]
            else:
                meta = self.setmetadata

            range = meta["rotation_range"]
            data = data[:, range[0] : range[1]]

        return data

    def load(self):
        with open(self.get_path(), "r") as f:
            data = np.array(json.load(f))
        return data

    def save(self, data):
        with open(self.get_path(), "w") as f:
            json.dump(data.tolist(), f)


class MotionDataset(IndividualfileDataset):
    pointclass = MotionDataPoint
