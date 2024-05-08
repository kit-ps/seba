from .abstract import AbstractMetric

import numpy
import scipy.stats as st


class DistanceMetric(AbstractMetric):
    """Metric for recognitions that compare original and modified data points.

    Expects that results have a single match - themselves in clear.

    Required pips:
        - numpy
        - scipy

    Parameters:
        none
    """

    def calc(self):
        identities = []
        dists = []

        for result in self.result.results:
            if result.identity not in identities:
                identities.append(result.identity)
            if numpy.isfinite(result.recognized[0]["dist"]):
                dists.append(result.recognized[0]["dist"])

        if st.sem(dists) == 0.0:
            conf = (numpy.mean(dists),)
        else:
            conf = st.t.interval(0.95, len(dists) - 1, loc=numpy.mean(dists), scale=st.sem(dists))

        return {
            "ids": len(identities),
            "n": len(dists),
            "avg": round(float(numpy.mean(dists)), 3),
            "std": round(float(numpy.std(dists)), 3),
            "min": round(float(numpy.min(dists)), 3),
            "max": round(float(numpy.max(dists)), 3),
            "conf": round(float(numpy.mean(dists) - conf[0]), 3),
        }
