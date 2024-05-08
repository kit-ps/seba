from .abstract import AbstractFaceDeanonymization

import cv2
import random
import numpy as np
import copy


class Learn_permutationDeanonymization(AbstractFaceDeanonymization):
    """De-Anonymize faces by learning a fixed permutation from the training set and then applying the reverse function

    NOTE: This will require that images are saved using a lossless format, e.g. PNG.

    Required pips:
        - opencv-python
        - numpy

    Parameters:
        - (int) max_img_b: Maximum images to analyze for black pixels (optional, default: 3)
        - (int) max_img_c: Maximum images to analyze for colored pixels (optional, default: 20)
    """

    name = "learn_permutation"

    def validate_config(self):
        if "max_img_b" not in self.config:
            self.config["max_img_b"] = 3
        else:
            self.config["max_img_b"] = int(self.config["max_img_b"])

        if "max_img_c" not in self.config:
            self.config["max_img_c"] = 20
        else:
            self.config["max_img_c"] = int(self.config["max_img_c"])

    def train(self, clear_set, anon_set):
        self.log.info("Learning permutation...")
        random.seed()
        perm = []
        i = 0

        for key, clear_point in clear_set.datapoints.items():
            if i > self.config["max_img_c"]:
                break

            self.log.info("Analyzing " + clear_point.pointname)
            unique = True
            clear_img = cv2.imread(clear_point.get_path())
            clear2d = self.reducedim(clear_img)
            anon_img = cv2.imread(anon_set.datapoints[key].get_path())
            anon2d = self.reducedim(anon_img)

            if not len(perm):
                perm = [[[] for y in range(len(clear2d[x]))] for x in range(len(clear2d))]

            for y in range(len(clear2d)):
                for x in range(len(clear2d[0])):
                    if len(perm[x][y]) == 1:
                        continue
                    if i >= self.config["max_img_b"] and clear2d[y][x] == 0:  # likely blackbar
                        unique = False
                        continue

                    p = np.where(anon2d == clear2d[y][x])
                    possible = [p[0][i] * 10000 + p[1][i] for i in range(len(p[0]))]

                    if not len(perm[x][y]):
                        perm[x][y] = possible  # first image
                    else:
                        perm[x][y] = np.intersect1d(perm[x][y], possible, assume_unique=True)

                    if len(perm[x][y]) == 0:
                        raise ValueError("Conflicting permutation noticed at {}, {}".format(x, y))
                    elif len(perm[x][y]) > 1:
                        unique = False

            if unique:
                self.log.info("Found unique correct permutation!")
                break
            i += 1

        if not unique:
            self.log.info("Could not determine unique permutation. Choosing a random possible one...")
            for x in range(len(perm)):
                for y in range(len(perm[x])):
                    if len(perm[x][y]) > 1:
                        random.shuffle(perm[x][y])
                        perm[x][y] = perm[x][y][:1]

                        for x2 in range(len(perm)):
                            for y2 in range(len(perm[x2])):
                                if x != x2 or y != y2:
                                    if not type(perm[x2][y2]) == list:
                                        perm[x2][y2] = list(perm[x2][y2])
                                    try:
                                        perm[x2][y2].remove(perm[x][y][0])
                                    except ValueError:
                                        pass

        self.inverse = [[[] for y in range(len(perm[x]))] for x in range(len(perm))]
        for x in range(len(perm)):
            for y in range(len(perm[x])):
                x2 = perm[x][y][0] // 10000
                y2 = perm[x][y][0] % 10000
                self.inverse[x2][y2] = (x, y)

    def reducedim(self, array):
        ra = [[[] for y in range(len(array[x]))] for x in range(len(array))]
        for x in range(len(array)):
            for y in range(len(array[0])):
                tmp = array[x, y][0]
                tmp += 256 * array[x, y][1]
                tmp += 256 * 256 * array[x, y][2]
                ra[x][y] = tmp
        return ra

    def deanonymize(self, image):
        orig_img = cv2.imread(image.get_path())
        img = copy.deepcopy(orig_img)

        for y in range(len(img)):
            for x in range(len(img[0])):
                img[self.inverse[x][y][1]][self.inverse[x][y][0]] = orig_img[x][y]

        cv2.imwrite(image.get_path(), img)
