from .abstract import AbstractMetric

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

class ConfusionMetric(AbstractMetric):
    """Generate a confusion matrix plot

    Required pips:
        - numpy
        - scipy

    Parameters:
        none
    """

    def calc(self):
        true = []
        predicted = []

        self.result.results.sort(key=lambda x: x.label)

        for result in self.result.to_squashed(strat="min").results:
            true.append(result.label)
            predicted.append(result.get_best_recognized()['label'])

        print(len(set(true)))
        print(len(set(predicted)))

        ConfusionMatrixDisplay.from_predictions(true, predicted, include_values=False, xticks_rotation='vertical', normalize="true")
        plt.show()
