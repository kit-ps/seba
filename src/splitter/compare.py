from .abstract import AbstractSplitter


class CompareSplitter(AbstractSplitter):
    name = "compare"
    random = False
    nin = 2
    nout = 2

    def split(self, in_sets):
        orig_set, anon_set = in_sets
        min_set = anon_set if len(anon_set.identities) <= len(orig_set.identities) else orig_set

        return [
            orig_set.copy(only_ids=min_set.identities.keys(), softlinked=True),
            anon_set.copy(only_ids=min_set.identities.keys(), softlinked=True),
        ]
