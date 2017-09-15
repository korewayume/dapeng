# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals
import os
import re
import numpy as np
from osgeo import gdal
from gdalconst import GA_ReadOnly
from keras_abc import Iterator
import keras.backend as K


def x_tif_reader(path):
    tif = gdal.Open(path, GA_ReadOnly)
    channels = []
    for band in [1, 2, 3]:
        channels.append(tif.GetRasterBand(band).ReadAsArray())
    return np.stack(channels, axis=-1).astype(K.floatx())


def y_tif_reader(path):
    tif = gdal.Open(path, GA_ReadOnly)
    channels = []
    for band in [1]:
        channels.append(tif.GetRasterBand(band).ReadAsArray())
    gray = np.stack(channels, axis=-1)
    gray[gray != 0] = 1
    return gray.astype(K.floatx())


class ImageDirectoryIterator(Iterator):
    def __init__(self, directory_x, directory_y,
                 x_reader, y_reader, filename_regex,
                 batch_size=32, shuffle=True, seed=None):
        self.x_reader = x_reader
        self.y_reader = y_reader

        x_filenames = [filename for filename in os.listdir(directory_x)
                       if re.match(filename_regex, filename)]
        y_filenames = [filename for filename in os.listdir(directory_y)
                       if re.match(filename_regex, filename)]

        assert set(x_filenames) == set(y_filenames), "x文件名与y文件名不一致。"

        filenames = sorted(x_filenames)

        self.path_pairs = [(os.path.join(directory_x, filename), os.path.join(directory_y, filename))
                           for filename in filenames]
        # 计算样本总数
        self.samples = len(filenames)

        print("总共找到{}个样本。".format(self.samples))

        super(ImageDirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batch_of_samples(self, index_array):
        batch_x, batch_y = [], []
        for i, j in enumerate(index_array):
            x_path, y_path = self.path_pairs[j]
            batch_x.append(self.x_reader(x_path))
            batch_y.append(self.y_reader(y_path))
        return np.stack(batch_x), np.stack(batch_y)


if __name__ == '__main__':
    dir_x, dir_y, bs = (u'/Users/mac/Desktop/beizhen/data/train/x',
                        u'/Users/mac/Desktop/beizhen/data/train/y',
                        10)
    gen = ImageDirectoryIterator(dir_x, dir_y, x_tif_reader, y_tif_reader, r"^\d{3}_\d{3}_\d{3}\.tif")
    print("{}batches.".format(len(gen)))
    for _ in range(10000):
        next(gen)
