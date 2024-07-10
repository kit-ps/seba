from .pcasvm import PcasvmClassification

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV


class LearnpcasvmClassification(PcasvmClassification):
    """Train and use a PCA + SVM. Learn params using BayesSearchCV.

    Based on https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_eigenfaces.html

    Required pips:
        sklearn, scikit-optimize

    Parameters:
        none
    """

    def enroll(self, set):
        pipeline = Pipeline([("scaler", StandardScaler()), ("pca", PCA()), ("svm", SVC())])
        params = {
            "pca__n_components": (20, 500, "uniform"),
            "pca__whiten": [True, False],
            "svm__C": (1e-6, 1e6, "log-uniform"),
            "svm__class_weight": [None, "balanced"],
            "svm__kernel": ["linear", "poly", "rbf"],
        }

        self.clf = BayesSearchCV(pipeline, params, n_iter=64)
        img, pred = self.load_data(set)
        self.clf.fit(img, pred)
