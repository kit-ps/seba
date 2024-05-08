from .abstract import AbstractFaceDeanonymization


class NoopDeanonymization(AbstractFaceDeanonymization):
    """Apply no deanonymization

    Required pips:
        none
    Parameters:
        none
    """

    name = "noop"
