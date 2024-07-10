from .abstract import AbstractMotionAnonymization


class NoopAnonymization(AbstractMotionAnonymization):
    """Apply no anonymization

    Required pips:
        none
    Parameters:
        none
    """

    name = "noop"

    def anonymize(self, mocap, data):
        return data
