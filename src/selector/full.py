from .abstract import AbstractSelector


class FullSelector(AbstractSelector):
    """Use the entire dataset

    Required pips:
        none

    Parameters:
        none
    """

    name = "full"

    def select(self, set):
        return set.copy(softlinked=True)
