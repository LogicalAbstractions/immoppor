import glob
import os
import ntpath
import cv2

import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from numpy import float32


def _iterate_files_(path: str, extension: str, recursive: bool = True):
    for file_path in glob.iglob(path + '/**/*.' + extension, recursive=recursive):
        yield file_path


class HeightDataset(Dataset):
    def __init__(self, root_path: str):
        self.__x_transform__ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.__y_transform__ = transforms.Compose([
            transforms.ToTensor()
        ])

        self.root_path = root_path
        sample_filenames = [x for x in _iterate_files_(
            root_path, 'jpg',
            True)]
        self.__sample_ids__ = [os.path.splitext(ntpath.basename(p))[0] for p in sample_filenames]

    def __len__(self):
        return len(self.__sample_ids__)

    def __getitem__(self, idx):
        x_path = os.path.join(self.root_path, self.__sample_ids__[idx] + '.jpg')
        y_path = os.path.join(self.root_path, self.__sample_ids__[idx] + ".png")

        x = self.__x_transform__(cv2.imread(x_path))
        y = self.__y_transform__(np.divide(cv2.imread(y_path, -1).astype(float32), 65536.0))

        return x, y