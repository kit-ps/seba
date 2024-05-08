from .abstract import AbstractFaceAnonymization


class NoopAnonymization(AbstractFaceAnonymization):
    """Apply no anonymization

    Required pips:
        none
    Parameters:
        none
    """

    name = "noop"
