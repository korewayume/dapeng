# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals
import os
import re
from functools import partial
import numpy as np
from keras_abc import Iterator
import keras.backend as K


def x_tif_reader(path):
    from osgeo import gdal
    from gdalconst import GA_ReadOnly
    tif = gdal.Open(path, GA_ReadOnly)
    channels = []
    for band in [1, 2, 3]:
        channels.append(tif.GetRasterBand(band).ReadAsArray())
    return np.stack(channels, axis=-1).astype(K.floatx())


def y_tif_reader(path):
    from osgeo import gdal
    from gdalconst import GA_ReadOnly
    tif = gdal.Open(path, GA_ReadOnly)
    channels = []
    for band in [1]:
        channels.append(tif.GetRasterBand(band).ReadAsArray())
    gray = np.stack(channels, axis=-1)
    gray[gray != 0] = 1
    return gray.astype(K.floatx())


def transform_image(image, angle=0, horizontal_flip=False, vertical_flip=False, gray=False):
    from skimage import transform
    dtype = image.dtype
    image = image.astype(np.uint8)
    if horizontal_flip:
        image = image[:, ::-1]
    if vertical_flip:
        image = image[::-1]
    h, w, d = image.shape

    center = (h * 3 / 2 - 0.5, w * 3 / 2 - 0.5)

    if gray:
        image = np.squeeze(image)
    transformed = transform.rotate(image, angle=angle, center=center, mode="reflect", preserve_range=True)
    if gray:
        transformed = np.expand_dims(transformed, -1)

    return transformed.astype(dtype)


def random_transform(rg):
    # angle = np.random.uniform(-rg, rg)
    angle = np.random.choice(np.random.uniform([-rg, -rg + 90, -rg + 180, -rg + 270], [rg, rg + 90, rg + 180, rg + 270]))
    vertical_flip = np.random.choice([True, False])
    horizontal_flip = np.random.choice([True, False])
    return partial(transform_image, angle=angle, vertical_flip=vertical_flip, horizontal_flip=horizontal_flip)


def predict_from_directory(directory_x, x_reader, filename_regex, batch_size=32):
    filenames = sorted([filename for filename in os.listdir(directory_x)
                        if re.match(filename_regex, filename)])
    path_pairs = [(os.path.join(directory_x, filename), None)
                  for filename in filenames]
    return ImageDirectoryIterator(path_pairs, x_reader, None, batch_size=batch_size, shuffle=False)


def train_from_directories(directory_x, directory_y,
                           x_reader, y_reader, filename_regex,
                           batch_size=32, shuffle=True, seed=None, random_transform_func=None):
    x_filenames = [filename for filename in os.listdir(directory_x)
                   if re.match(filename_regex, filename)]
    y_filenames = [filename for filename in os.listdir(directory_y)
                   if re.match(filename_regex, filename)]

    assert set(x_filenames) == set(y_filenames), "x文件名与y文件名不一致。"

    filenames = sorted(x_filenames)

    path_pairs = [(os.path.join(directory_x, filename), os.path.join(directory_y, filename))
                  for filename in filenames]

    return ImageDirectoryIterator(path_pairs, x_reader, y_reader, batch_size=batch_size, shuffle=shuffle, seed=seed,
                                  random_transform_func=random_transform_func)


class ImageDirectoryIterator(Iterator):
    def __init__(self, path_pairs, x_reader=None, y_reader=None,
                 batch_size=32, shuffle=True, random_transform_func=None, seed=None):
        self.x_reader = x_reader
        self.y_reader = y_reader
        self.path_pairs = path_pairs
        self.random_transform_func = random_transform_func
        # print("总共找到{}个样本。".format(len(self.path_pairs)))
        super(ImageDirectoryIterator, self).__init__(len(self.path_pairs), batch_size, shuffle, seed)

    def _get_batch_of_samples(self, index_array):
        batch_x, batch_y = [], []
        for i in index_array:
            x_path, y_path = self.path_pairs[i]
            x = self.x_reader(x_path)
            x = self.random_transform_func(x) if self.random_transform_func else x
            batch_x.append(x)
            if self.y_reader:
                y = self.y_reader(y_path)
                y = self.random_transform_func(y, gray=True) if self.random_transform_func else y
                batch_y.append(y)
        if batch_y:
            return np.stack(batch_x), np.stack(batch_y)
        else:
            return np.stack(batch_x)


if __name__ == '__main__':
    dir_x, dir_y, bs = (u'/Users/mac/Desktop/beizhen/data/train/x',
                        u'/Users/mac/Desktop/beizhen/data/train/y',
                        10)
    gen = train_from_directories(dir_x, dir_y, x_tif_reader, y_tif_reader, r"^\d{3}_\d{3}_\d{3}\.tif")
    print("{}batches.".format(len(gen)))
    for _ in range(10000):
        next(gen)
