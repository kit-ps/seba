from .abstract import AbstractMotionAnonymization

from sklearn.preprocessing import normalize


class NormalizeAnonymization(AbstractMotionAnonymization):
    """Normalizes each mocap point between 0 and 1

    Required pips:
        sklearn
    Parameters:
        none
    """

    name = "normalize"

    def anonymize(self, mocap, data):
        new_data = normalize(data, axis=0)

        return new_data
