from .abstract import AbstractSelector

from sklearn.preprocessing import minmax_scale
import numpy as np


class GreedySelector(AbstractSelector):
    """Select identities based on meta attributes using a greedy algorithm to find a diverse set of identities
        Test each identity as a starting point and use the one that creates the most diverse set.
        During each iteration of building a set, the next identity has the maximum minimum euclidean distance to the current ids in the set.

    Required pips:
        sklearn, numpy

    Parameters:
        - (int) ids: number of identities to select (required)
        - (list[string]) attributes: list of strings with given attributes that are compared, ["sex", "age", "mass", "height"] (required)
        - (dict[string:float]) encoding: dict that gives the encoding of string values {"female": 1, "male": 0} (optional, default: {})
    """

    name = "greedy"

    def select(self, set):
        # Scaling of the attribute vectors to a value between 0 and 1
        ids = list(set.identities.keys())
        feature_vectors = []
        for i in ids:
            tmp_in = []
            for att in self.config["attributes"]:
                value = getattr(set.identities[i], att)
                if type(value) is list:
                    tmp_in = tmp_in + value
                else:
                    if value in self.config["encoding"]:
                        value = self.config["encoding"][value]
                    tmp_in.append(value)
            feature_vectors.append(tmp_in)
        feature_vectors = minmax_scale(feature_vectors)

        # Calculation of the distance from each subject to each other subject O(n²)
        distance_matrix = []
        for i in feature_vectors:
            row = []
            for ii in feature_vectors:
                row.append(np.linalg.norm(i - ii))
            distance_matrix.append(row)

        distance_matrix = np.array(distance_matrix)

        # Greedy building of the set with each identity as start point.
        current_distance = 0
        min_dis_set = []
        for k in range(len(ids)):
            tmp_set = [k]
            for i in range(self.config["ids"] - 1):
                ixgrid = np.ix_(tmp_set)
                foo = distance_matrix[ixgrid]

                if i == 0:
                    min_distance = foo.max(axis=1)
                    selection = foo.argmax(axis=1)[0]
                else:
                    min_vec = foo.min(axis=0)
                    min_distance = min_vec.max()
                    selection = min_vec.argmax()
                tmp_set.append(selection)

            if min_distance > current_distance:
                min_dis_set = tmp_set
                current_distance = min_distance

        only_ids = []
        for k in min_dis_set:
            only_ids.append(ids[k])

        return set.copy(only_ids=only_ids, softlinked=True)
