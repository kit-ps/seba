from .abstract import AbstractFaceAnonymization

import cv2
import random
import copy


class BlockpermutateAnonymization(AbstractFaceAnonymization):
    """Apply a block permutation to the face in an image

    Required pips:
        - opencv-python

    Parameters:
        - (string) seed: seed for the permutation
        - (int) blocksize: size of blocks to permutate
    """

    name = "blockpermutate"

    def validate_config(self):
        if "seed" not in self.config:
            raise AttributeError("BlockPermutateAnonymization: config: missing seed")

        if "blocksize" not in self.config:
            raise AttributeError("BlockPermutateAnonymization: config: missing blocksize")
        else:
            self.config["blocksize"] = int(self.config["blocksize"])

    def anonymize(self, image):
        random.seed(a=self.config["seed"])

        img = cv2.imread(image.get_path())
        newimg = copy.deepcopy(img)

        blocks_x = img.shape[0] // self.config["blocksize"]
        blocks_y = img.shape[1] // self.config["blocksize"]

        permutation = [(y, x) for x in range(blocks_x) for y in range(blocks_y)]
        random.shuffle(permutation)

        for x in range(blocks_x):
            for y in range(blocks_y):
                new_y, new_x = permutation.pop()
                newimg[
                    (y * self.config["blocksize"]) : ((y + 1) * self.config["blocksize"]),
                    (x * self.config["blocksize"]) : ((x + 1) * self.config["blocksize"]),
                ] = img[
                    (new_y * self.config["blocksize"]) : ((new_y + 1) * self.config["blocksize"]),
                    (new_x * self.config["blocksize"]) : ((new_x + 1) * self.config["blocksize"]),
                ]

        cv2.imwrite(image.get_path(), newimg)
