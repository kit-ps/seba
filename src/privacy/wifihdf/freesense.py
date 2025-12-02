from ...lib.result import Result
from .abstract import AbstractWifihdfPrivacy
from ...lib.inference import Classification

import math
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pywt
import dtw

class FreesensePrivacy(Classification, AbstractWifihdfPrivacy):
    """Uses the system by Xin et al. from "FreeSense:Indoor Human Identification with WiFi Signals"

    Required pips:
        - pytorch

    Parameters:
        None
    """

    def validate_config(self):
        for x in ['length', 'bt_order', 'bt_freq', 'antennas', 'knn_n', 'pca_n']:
            if x not in self.config:
                raise AttributeError("Configuration for FreeSense Classification is missing parameter {}!".format(x))

    def preprocess(self, dataset):
        for point in dataset.points:
            point.data = self.preprocess_point(point.data)

    def preprocess_point(self, data):
        rs = []
        key = 'flattened_mags'
        for i in range(data[key].shape[1]):
            data[key][:, i] = self.butterworth_filter(data[key][:, i])
        data[key] = data[key].reshape(data[key].shape[0], self.config['antennas'], -1)
        for i in range(self.config['antennas']):
            at = []
            pdt = self.pca(data[key][:, i, :])
            for pd in pdt:
                dwt = self.dwt(pd)
                at.append(dwt)
            rs.append(at)
        return np.array(rs).flatten()

    def dwt(self, data):
        cA, _ = pywt.dwt(data, 'db4')
        return cA

    def pca(self, data):
        pca = PCA(self.config['pca_n'])
        pdt = pca.fit_transform(data).transpose()
        return sorted(list(pdt), reverse=True, key=lambda x: np.ptp(x))

    def butterworth_filter(self, sequence):
        csi_samplerate = self.config['bt_freq']
        walk_frequncy = 10
        w_c = (2 * math.pi * walk_frequncy) / (csi_samplerate)
        b, a = signal.butter(self.config['bt_order'], w_c, analog=True)
        filtered_seq = signal.lfilter(b, a, sequence)
        return filtered_seq

    def enroll(self, dataset):
        # make data length the same for all and remove timestamp-delta
        dataset.cut_or_pad(self.config['length'])
        dataset.remove_ts()

        self.log.info("Pre-processing enrollment dataset")
        self.preprocess(dataset)

        self.log.info("Fitting k-NN")
        self.knn = KNeighborsClassifier(
            n_neighbors=self.config['knn_n'],
            metric=freesense_distance,
            metric_params={'antennas': self.config['antennas'], 'pca_n': self.config['pca_n']},
            n_jobs=-1
        )
        self.knn.fit(
            list(map(lambda x: x.data, dataset.points)),
            list(map(lambda x: x.label, dataset.points))
        )

    def classify_all(self, dataset, resultset):
        dataset.cut_or_pad(self.config['length'])
        dataset.remove_ts()
        self.log.info("Pre-processing test dataset")
        self.preprocess(dataset)

        self.log.info("Classifying dataset")
        for point in dataset.points:
            rs = Result(point.label, point.pointname)
            results = np.average(self.knn.predict_proba(point.data.reshape(1, -1)), axis=0)
            for i in range(len(results)):
                rs.add_recognized(self.knn.classes_[i], dist=(1 - results[i]))
            resultset.append(rs)
        return resultset


def freesense_distance(x, y, **kwargs):
    sum = 0.0
    x = x.reshape(kwargs['antennas'], kwargs['pca_n'], -1)
    y = y.reshape(kwargs['antennas'], kwargs['pca_n'], -1)
    for i in range(len(x)):
        for j in range(len(x[i])):
            sum += dtw.dtw(x[i][j], y[i][j], distance_only=True).distance
    return sum
