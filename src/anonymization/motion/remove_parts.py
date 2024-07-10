from .abstract import AbstractMotionAnonymization


class Remove_partsAnonymization(AbstractMotionAnonymization):
    """Remove specific body parts from the data

    Required pips:
        none
    Parameters:
        (string) body_parts: Body parts to remove/kept as list. Possible elements ["hip", "left_leg", "right_leg", "torso", "left_arm", "head", "right_arm"]
        (bool) invert: Instead of removing the give points, only the give points are kept
    """

    name = "remove_parts"

    def validate_config(self):
        if "body_parts_to_marker" not in self.dataset.meta["original_meta"]:
            self.log.error("Missing body parts to marker meta information")
        if "body_parts" not in self.config:
            self.log.error("Missing body parts to remove")
        if "invert" not in self.config:
            self.log.info("Missing invert value for remove body part, taking default FALSE")
            self.config["invert"] = False

    def anonymize(self, mocap, data):
        marker_sequence = []
        if not self.config["invert"]:
            for part in self.config["body_parts"]:
                marker_sequence = marker_sequence + self.dataset.meta["original_meta"]["body_parts_to_marker"][part]
        else:
            for part in self.dataset.meta["original_meta"]["body_parts_to_marker"]:
                if part not in self.config["body_parts"]:
                    marker_sequence = marker_sequence + self.dataset.meta["original_meta"]["body_parts_to_marker"][part]

        array_index = []
        for i in marker_sequence:
            array_index.append(i * 3)
            array_index.append(i * 3 + 1)
            array_index.append(i * 3 + 2)

        for i in range(len(data)):
            for k in array_index:
                data[i][k] = 0

        return data
