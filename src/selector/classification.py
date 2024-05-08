from .abstract import AbstractSelector


class ClassificationSelector(AbstractSelector):
    """Use the identities with highest classification accuracy in the set

    Required pips:
        none

    Parameters:
        - (int) ids: number of identities to select
        - (string) key: attribute on identity-level to sort by (optional,, default: classification_hitrate)
    """

    name = "classification"
    random = False

    def select(self, set):
        ids = list(set.identities.keys())
        self.config["key"] = "classification_hitrate" if "key" not in self.config else self.config["key"]

        tmp = []
        for i in range(len(ids)):
            tmp.append((i, getattr(set.identities[ids[i]], self.config["key"])))

        tmp.sort(key=lambda y: y[1])

        tmp = tmp[-1 * self.config["ids"] :]

        only_ids = []
        for k, _ in tmp:
            only_ids.append(ids[k])

        return set.copy(only_ids=only_ids, softlinked=True)
