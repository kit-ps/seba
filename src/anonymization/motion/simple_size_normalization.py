from .abstract import AbstractMotionAnonymization


class Simple_size_normalizationAnonymization(AbstractMotionAnonymization):
    """Apply no anonymization

    Required pips:
        none
    Parameters:
        (dict) body_parts_to_marker: Must be given in the metadata of the dataset
        (int) axis: Specify axis to normalize, otherwise all axes are normalized
    """

    name = "simple_size_normalization"

    def validate_config(self):
        if "normalized_height" not in self.config:
            self.log.info("Missing normalized height, taking 1.80")
            self.config["normalized_height"] = 1.8

    def anonymize(self, mocap, data):
        id = mocap.idname

        height = self.dataset.identities[id].height

        scaling_factor = self.config["normalized_height"] / height

        if "axis" in self.config:
            for pose in data:
                pose[self.config["axis"] - 1 :: 3] *= scaling_factor
        else:
            data = data * scaling_factor

        return data
