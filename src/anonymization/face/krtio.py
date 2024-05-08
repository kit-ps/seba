from .abstract import AbstractFaceAnonymization
from ...lib.data.set import Dataset

import cv2
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import copy


class KrtioAnonymization(AbstractFaceAnonymization):
    """Apply a K-RTIO anonymization

    Paper: A. Rajabi, R. B. Bobba, M. Rosulek, C. V. Wright, and W. Feng,
    “On the (Im)Practicality of Adversarial Perturbation for Image Privacy,”
    Proceedings on Privacy Enhancing Technologies, vol. 2021, no. 1, pp. 85–106, Jan. 2021, doi: 10.2478/popets-2021-0006.

    Required pips:
        - opencv-python
        - pycryptodome

    Parameters:
        - (int) key: key as seed for pseudorandom-function (must be convertible to 16 bytes)
        - (string) overlay: dataset that includes the images to use as overlay images (optional, seed: krtio-overlay)
        - (float) alpha: alpha when overlaying the images (optional, default 0.5)
        - (int) blocksize: size of blocks when permutating overlay images (optional, default 60)
        - (int) k: number of images from overlay set to choose for overlay (optional, default 3)
    """

    name = "krtio"

    def validate_config(self):
        if "key" not in self.config:
            raise AttributeError("requires key (integer convertable to 16 bytes)")

        if "overlay" not in self.config:
            self.config["overlay"] = "krtio-overlay"

        if "alpha" not in self.config:
            self.config["alpha"] = 0.5
        else:
            self.config["alpha"] = float(self.config["alpha"])

        if "blocksize" not in self.config:
            self.config["blocksize"] = 60
        else:
            self.config["blocksize"] = int(self.config["blocksize"])

        if "k" not in self.config:
            self.config["k"] = 3
        else:
            self.config["k"] = int(self.config["k"])

    def init(self):
        self.overlays = Dataset(self.config["overlay"]).datapoints
        self.cipher = AES.new(self.config["key"].to_bytes(16, "big"), AES.MODE_CBC)

    def shuffle(self, x, key):
        for i in range(len(x) - 1, 0, -1):
            j = self.randrange(i + 1, key + str(i))
            x[i], x[j] = x[j], x[i]
        return x

    def randrange(self, max, seed):
        input = pad(seed.encode("utf-8"), AES.block_size)
        x = self.cipher.encrypt(input)
        return int.from_bytes(x, "big") % max

    def anonymize(self, image):
        aimg = cv2.imread(image.get_path())

        overlay_pixels = []
        for j in range(self.config["k"]):
            index = self.randrange(len(self.overlays), image.pointname + "0" + str(j))
            overlay = list(self.overlays.values())[index]
            img = cv2.imread(overlay.get_path())
            y_blocks = int(len(img) / self.config["blocksize"])
            x_blocks = int(len(img[0]) / self.config["blocksize"])
            permutation = [(y, x) for x in range(x_blocks) for y in range(y_blocks)]
            self.shuffle(permutation, image.pointname + "1" + str(j))
            ov_pixel = copy.deepcopy(img)
            for x in range(x_blocks):
                for y in range(y_blocks):
                    new_y, new_x = permutation.pop()
                    ov_pixel[
                        (y * self.config["blocksize"]) : ((y + 1) * self.config["blocksize"]),
                        (x * self.config["blocksize"]) : ((x + 1) * self.config["blocksize"]),
                    ] = img[
                        (new_y * self.config["blocksize"]) : ((new_y + 1) * self.config["blocksize"]),
                        (new_x * self.config["blocksize"]) : ((new_x + 1) * self.config["blocksize"]),
                    ]
            overlay_pixels.append(ov_pixel)

        for y in range(aimg.shape[0]):
            for x in range(aimg.shape[1]):
                for i in range(3):
                    aimg[y, x][i] = self.config["alpha"] * aimg[y, x][i] + (1 - self.config["alpha"]) * (1 / self.config["k"]) * sum(
                        map(lambda ov: ov[y, x][i], overlay_pixels)
                    )

        cv2.imwrite(image.get_path(), aimg)
