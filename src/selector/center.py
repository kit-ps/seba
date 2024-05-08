from .abstract import AbstractSelector

from sklearn.preprocessing import minmax_scale
import numpy as np


class CenterSelector(AbstractSelector):
    """Select identities based on meta attributes using a average-based algorithm to find a diverse set of identities
        Test each pair of identities as starting points and use the ones that create the most diverse set.
        During each iteration of building a set, the next identity has the maximum euclidean distance to the center of current identities.

    Required pips:
        sklearn, numpy

    Parameters:
        - (int) ids: number of identities to select (required)
        - (list[string]) attributes: list of strings with given attributes that are compared, ["sex", "age", "mass", "height"] (required)
        - (dict[string:float]) encoding: dict that gives the encoding of string values {"female": 1, "male": 0} (optional, default: {})
    """

    name = "center"

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

        selected_set_min_max = 0
        for i in range(len(feature_vectors)):
            current_set = self.create_set(i, feature_vectors)

            min_dist = 100000
            for k in range(len(current_set)):
                for kk in range(k + 1, len(current_set)):
                    tmp_min = distance_matrix[current_set[k]][current_set[kk]]
                    if tmp_min <= min_dist:
                        min_dist = tmp_min

            if tmp_min > selected_set_min_max:
                selected_set_min_max = tmp_min
                selected_set = current_set

        only_ids = []
        for k in selected_set:
            only_ids.append(ids[k])

        return set.copy(only_ids=only_ids, softlinked=True)

    # Starts with one feature vector and always adds the feature vector which is the furthest away from the center of
    # already selected feature vectors. Returns the index list of the selected feature vectors.
    def create_set(self, first_index, tmp_vecs):
        current_set_index = [first_index]
        current_set_vec = [tmp_vecs[first_index]]
        average = tmp_vecs[first_index]
        for _ in range(1, self.config["ids"]):
            max = 0
            max_index = 0
            for i in range(len(tmp_vecs)):
                if i not in current_set_index:
                    tmp = np.linalg.norm(average - tmp_vecs[i])
                    if tmp > max:
                        max = tmp
                        max_index = i

            current_set_index.append(max_index)
            current_set_vec.append(tmp_vecs[max_index])
            average = np.average(current_set_vec, axis=0)
        return current_set_index
