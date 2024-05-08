import torch
from torch.utils.data import Dataset

import cv2
import numpy as np
import random
from ..data.point import Image, PNGImage


class TorchImageDataset(Dataset):
    def __init__(self, filenames, transform=None):
        self.filenames = filenames
        self.len = len(filenames)
        self.transform = transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = cv2.imread(filename)
        img.astype(float)
        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)  # shape: width, height, channels -> channels, width, height

        if self.transform:
            img = self.transform(img)
        return img.float()

    def split(self, rate):
        threshold = int(self.len * rate)
        return (
            TorchImageDataset(self.filenames[:threshold], self.transform),
            TorchImageDataset(self.filenames[threshold:], self.transform),
        )

    def __len__(self):
        return self.len

    def update(self, index, new):
        new = new.detach().cpu().numpy()
        new = np.transpose(new, (1, 2, 0))
        new = new * 255
        cv2.imwrite(self.filenames[index], new)

    def shuffle(self, seed=None):
        random.seed(a=seed)
        random.shuffle(self.filenames)

    @staticmethod
    def from_set(set):
        points = list(set.datapoints.values())
        if not type(points[0]) in [Image, PNGImage]:
            raise AttributeError("TorchImageDataset: Datapoints must be images")

        filenames = list(map(lambda x: x.get_path(), points))
        return TorchImageDataset(filenames)


class TupleDataset(Dataset):
    def __init__(self, set1, set2):
        self.set1 = set1
        self.set2 = set2

        if not len(set1) == len(set2):
            raise ValueError("the two sets must be same length")
        self.len = len(set1)

    def __getitem__(self, index):
        return (self.set1[index], self.set2[index])

    def __len__(self):
        return self.len

    def split(self, rate):
        s1a, s1b = self.set1.split(rate)
        s2a, s2b = self.set2.split(rate)
        return (TupleDataset(s1a, s2a), TupleDataset(s1b, s2b))

    def shuffle(self, seed="seed"):
        self.set1.shuffle(seed)
        self.set2.shuffle(seed)
