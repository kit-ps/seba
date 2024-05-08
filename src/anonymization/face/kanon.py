from .abstract import AbstractFaceAnonymization

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import cv2


class KanonAnonymization(AbstractFaceAnonymization):
    """Abstract class for k-anonymity based face anonymizations

    Required pips:
        - sklearn
        - numpy
        - opencv

    Parameters:
        - (int) k: number of similar identities to choose (-1)
        - (int) pcan: number of components of the PCA
    """

    name = "kanon"

    def validate_config(self):
        if "k" not in self.config:
            raise AttributeError("K-Anonymity anonymization missing parameter k")

        if "pcan" not in self.config:
            raise AttributeError("K-Anonymity anonymization missing parameter pcan")

    def anonymize_all(self):
        if not self.bg:
            raise AttributeError("K-Anonymity anonymization requires background data set. Add a anonbg_rate parameter to your config!")

        images = []
        for point in self.bg.datapoints.values():
            img = cv2.imread(point.get_path())
            images.append(img.flatten())
        images = np.array(images)

        self.log.debug("Fitting Scaler")
        self.scaler = StandardScaler()
        normed_imgs = self.scaler.fit_transform(images)

        self.log.debug("Fitting PCA")
        self.pca = PCA(n_components=self.config["pcan"])
        self.features = self.pca.fit_transform(normed_imgs)
        self.log.debug("Finished Setup")

        for point in self.dataset.datapoints.values():
            img = cv2.imread(point.get_path())
            f = self.pca.transform(self.scaler.transform(img.flatten().reshape(1, -1)))

            distances = []
            for feat in self.features:
                distances.append(self.findCosineDistance(f[0], feat))

            x = np.argsort(distances)
            picked_ids = []
            picked_imgs = []
            i = 0

            while len(picked_imgs) < (self.config["k"] - 1):
                if len(x) <= i:
                    break
                p = list(self.bg.datapoints.values())[x[i]]
                if p.idname not in picked_ids:
                    picked_ids.append(p.idname)
                    picked_imgs.append(p)
                i += 1

            picked_imgs.append(point)
            img = self.merge_images(picked_imgs, shape=img.shape)
            cv2.imwrite(point.get_path(), img)

    def findCosineDistance(self, x, y):
        a = np.matmul(np.transpose(x), y)
        b = np.sum(np.multiply(x, x))
        c = np.sum(np.multiply(y, y))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def merge_images(self, points, shape=()):
        raise NotImplementedError("Don't use the abstract class")
