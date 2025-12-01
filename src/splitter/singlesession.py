from .abstract import AbstractSplitter

class SinglesessionSplitter(AbstractSplitter):
    """Simple splitter to select a single session

    Required pips:
        none

    Parameters:
        (string) target_session: the name of the single session to select
    """

    name = "singlesession"
    random = False
    nin = 1
    nout = 1

    def validate_config(self):
        if "target_session" not in self.config:
            raise AttributeError("Splitter: Missing target session")
        else:
            self.target = self.config["target_session"]

    def split(self, in_sets):
        return [in_sets[0].copy(
            sess_filter=(lambda x: int(x.session) == self.target),
            softlinked=True
        )]
