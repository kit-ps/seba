from .abstract import AbstractMotionPrivacy
from ...lib.inference import Classification
from ...lib.result import Result

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import numpy as np


def simple_feature_vector(data):
    pca = PCA(n_components=4)
    pca.fit(data)
    eigen_postures = pca.components_
    average_posture = data.mean(axis=0)

    return np.concatenate((eigen_postures[0], average_posture), axis=None)


class SklearnClassification(Classification, AbstractMotionPrivacy):
    """Use the scikit-learn machine learning methods.
    Sklearn documentation: https://scikit-learn.org/stable/index.html

    Required pips:
        - sklearn

    Parameters:
        - (string) feature_rep: simple, segment
        - (string) classifier: classifier name for the sklearn classifier: svm, randomForest, decisionTree, kNeighbors
        - (int)    number_of_splits: How many splits to perform for the cross validation
        - {string) attribute: The attribute to classify, the default is "identity". Can be any attribute in the identity metadata
        - (dict)   attribute_conversion: A dict that maps none number values into numbers
    """

    def __init__(self, config):
        super().__init__(config)
        self.train_labels = None
        self.train_set = None
        self.set_metadata = None
        self.trained_clf = None
        self.clf_init = {
            "svm": SVC(probability=True),
            "randomForest": RandomForestClassifier(n_estimators=10),
            "decisionTree": DecisionTreeClassifier(random_state=0),
            "kNeighbors": KNeighborsClassifier(n_neighbors=3),
            "gNaiveBayes": GaussianNB(),
        }

    # This feature representation was taken from "Personal identifiability of user tracking data during observation of 360-degree VR video" by Miller et. al.
    def segment_base_vector(self, data, meta):
        body_to_markers = meta["body_parts_to_position"]
        fps = meta["fps"]

        new_poses = []
        for i in range(len(data)):
            new_pose = []
            new_poses.append(new_pose)
            for part in body_to_markers:
                seq = body_to_markers[part]
                xs = []
                ys = []
                zs = []
                for ele in seq:
                    xs.append(data[i][ele * 3])
                    ys.append(data[i][ele * 3 + 1])
                    zs.append(data[i][ele * 3 + 2])

                new_pose.append(np.average(np.array(xs)))
                new_pose.append(np.average(np.array(ys)))
                new_pose.append(np.average(np.array(zs)))

        new_poses = np.array(new_poses)
        feature_vecs = []
        pos = 0
        while pos + fps < len(new_poses):
            max = np.max(new_poses[pos : pos + fps], axis=0)
            min = np.min(new_poses[pos : pos + fps], axis=0)
            median = np.median(new_poses[pos : pos + fps], axis=0)
            mean = np.mean(new_poses[pos : pos + fps], axis=0)
            std = np.std(new_poses[pos : pos + fps], axis=0)
            vec = np.concatenate((max, min, median, mean, std))
            feature_vecs.append(vec)
            pos = pos + fps

        return feature_vecs

    def validate_config(self):
        if "feature_rep" not in self.config:
            self.config["feature_rep"] = "simple"
            self.log.info("Missing parameter value for feature_rep, selecting default value 'simple'")

        if "classifier" not in self.config:
            self.config["classifier"] = "svm"
            self.log.info("Missing parameter value for classifier, selecting default value 'svm'")

        if "attribute" not in self.config:
            self.config["attribute"] = "identity"
            self.log.info("Missing parameter value for attribute, selecting default value 'identity'")

        if "attribute_conversion_type" not in self.config:
            self.config["attribute_conversion_type"] = "dict"
            self.log.info("Missing parameter value for attribute_conversion_type, selecting default value 'dict'")

        if "number_of_splits" not in self.config:
            self.config["number_of_splits"] = 10
            self.log.info("Missing parameter value for number_of_splits, selecting default value 10'")

    def enroll(self, set):
        self.log.info("Starting privacy.\n\tConfiguration: " + str(self.config))

        if self.config["attribute"] == "identity":
            self.train_labels = [e.idname for e in set.datapoints.values()]
        else:
            self.train_labels = []
            for e in set.datapoints.values():
                label = getattr(e.identity, self.config["attribute"])
                label = self.config["attribute_conversion"][label]
                self.train_labels.append(label)

        self.train_set = np.array([e.load() for e in list(set.datapoints.values())])
        self.set_metadata = set.meta["original_meta"]

        feature_vecs = []
        labels = []
        if self.config["feature_rep"] == "none":
            for i in range(len(self.train_set)):
                feature_vecs.append(self.train_set[i])
                labels.append(self.train_labels[i])

        if self.config["feature_rep"] == "flatten":
            for i in range(len(self.train_set)):
                feature_vecs.append(self.train_set[i].flatten())
                labels.append(self.train_labels[i])

        if self.config["feature_rep"] == "simple":
            for i in range(len(self.train_set)):
                feature_vecs.append(simple_feature_vector(self.train_set[i]))
                labels.append(self.train_labels[i])

        if self.config["feature_rep"] == "segment":
            for i in range(len(self.train_set)):
                tmp = self.segment_base_vector(self.train_set[i], self.set_metadata)
                feature_vecs = feature_vecs + tmp
                labels = labels + [self.train_labels[i]] * len(tmp)

        clf = self.clf_init[self.config["classifier"]]

        k_fold_strat = StratifiedKFold(n_splits=self.config["number_of_splits"])
        pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA()), ("clf", clf)])
        cv_results = cross_validate(pipe, feature_vecs, labels, cv=k_fold_strat, scoring="balanced_accuracy", return_estimator=True)
        max_i = cv_results["test_score"].argmax(axis=0)
        self.log.info("Validation score: " + str(cv_results["test_score"][max_i]))
        self.trained_clf = cv_results["estimator"][max_i]

    def classify_point(self, mocap):
        data = mocap.load()

        if self.config["attribute"] == "identity":
            label = mocap.idname
        else:
            label = getattr(mocap.identity, self.config["attribute"])

        if self.config["feature_rep"] == "simple":
            feature_vec = simple_feature_vector(data).reshape(1, -1)
            results = self.trained_clf.predict_proba(feature_vec)[0]
        if self.config["feature_rep"] == "segment":
            feature_vec = self.segment_base_vector(data, self.set_metadata)
            results = np.average(self.trained_clf.predict_proba(feature_vec), axis=0)
        if self.config["feature_rep"] == "none":
            results = self.trained_clf.predict_proba(data.reshape(1, -1))[0]
        if self.config["feature_rep"] == "flatten":
            feature_vec = (data.flatten()).reshape(1, -1)
            results = self.trained_clf.predict_proba(feature_vec)[0]

        rs = Result(label, mocap.pointname)

        if self.config["attribute"] == "identity":
            for i in range(len(results)):
                rs.add_recognized(self.trained_clf.classes_[i], dist=1 - results[i])
        else:
            reverse_attribute_conversion = {v: k for k, v in self.config["attribute_conversion"].items()}
            for i in range(len(results)):
                rs.add_recognized(reverse_attribute_conversion[self.trained_clf.classes_[i]], dist=1 - results[i])

        return rs

    def cleanup(self):
        pass
