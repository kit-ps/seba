from .abstract import AbstractSelector


class ZooSelector(AbstractSelector):
    """Use the genuine and imposter scores of every identity to sort them

    Required pips:
        none

    Parameters:
        - (int) ids: number of identities to select
        - (int) genuine: weight of the genuine score
        - (int) imposter: weight of the imposter score
        - (bool) use_eucl: use the euclidean distance instead of cosine similarity
    """

    name = "zoo"

    def select(self, set):
        key = "_score_cos"
        if self.config["use_eucl"]:
            key = "_score_eucl"

        ids = list(map(lambda x: (x.name, (-1) * getattr(x, "gen" + key)["max"], getattr(x, "imp" + key)["min"]), set.identities.values()))
        ids = list(map(lambda x: (x[0], (self.config["genuine"] * x[1] + self.config["imposter"] * x[2])), ids))

        ids.sort(key=lambda x: x[1], reverse=True)
        only_ids = list(map(lambda x: x[0], ids))[: self.config["ids"]]
        return set.copy(only_ids=only_ids, softlinked=True)
