from .abstract import AbstractMotionAnonymization


class Sum_of_partsAnonymization(AbstractMotionAnonymization):
    """Apply no anonymization

    Required pips:
        none
    Parameters:
        (dict) body_parts_to_marker: Must be given in the metadata of the dataset
    """

    name = "sum_of_parts"

    def validate_config(self):
        if "body_parts_to_marker" not in self.dataset.meta["original_meta"]:
            self.log.error("Missing body parts to marker meta information")

    def anonymize(self, mocap, data):
        def translate_points(pose, points_index, translation_vector):
            for i in points_index:
                pose[3 * i] = pose[3 * i] + translation_vector[0]
                pose[3 * i + 1] = pose[3 * i + 1] + translation_vector[1]
                pose[3 * i + 2] = pose[3 * i + 2] + translation_vector[2]

        body_to_marker = self.dataset.meta["original_meta"]["body_parts_to_marker"]

        for pose in data:
            translate_points(pose, body_to_marker["right_leg"], [0, 500, 1000])
            translate_points(pose, body_to_marker["left_leg"], [0, -500, 0])
            translate_points(pose, body_to_marker["torso"], [0, -500, -500])
            translate_points(pose, body_to_marker["head"], [0, 500, -1000])
            translate_points(pose, body_to_marker["left_arm"], [0, -500, 500])
            translate_points(pose, body_to_marker["right_arm"], [0, 500, -500])

        return data
