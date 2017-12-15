# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
from functools import partial
from keras.models import Model
from keras.optimizers import Adam
from dapeng.metrics import jaccard_coefficient
from keras.regularizers import l1
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, UpSampling2D, Dropout, Input, concatenate

"""
from keras.optimizers import Adam
from dapeng.mertrics import jaccard_coefficient
model = unet5(256, optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=[jaccard_coefficient])
"""

model_weights_archive = os.path.expanduser("~/.dapeng/weights/")

if not os.path.exists(model_weights_archive):
    os.makedirs(model_weights_archive)


CommonConv2D = partial(
    Conv2D,
    activation='relu',
    padding='same',
    kernel_regularizer=l1(1e-3),
    bias_regularizer=l1(1e-3),
    activity_regularizer=l1(1e-3)
)


def unet5(size, optimizer=None, loss=None, metrics=None, dropout_rate=0.2, load_weights=False):

    if optimizer is None:
        optimizer = Adam(lr=1e-3)
    if loss is None:
        loss = "binary_crossentropy"
    if metrics is None:
        metrics = [jaccard_coefficient]

    inputs = Input(shape=(size, size, 3))
    # left

    conv1 = CommonConv2D(32, (3, 3))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = CommonConv2D(32, (3, 3))(conv1)
    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    conv2 = CommonConv2D(64, (3, 3))(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = CommonConv2D(64, (3, 3))(conv2)
    conv2 = BatchNormalization()(conv2)

    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    conv3 = CommonConv2D(128, (3, 3))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = CommonConv2D(128, (3, 3))(conv3)
    conv3 = BatchNormalization()(conv3)

    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    conv4 = CommonConv2D(256, (3, 3))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = CommonConv2D(256, (3, 3))(conv4)
    conv4 = BatchNormalization()(conv4)

    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
    conv5 = CommonConv2D(512, (3, 3))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = CommonConv2D(512, (3, 3))(conv5)
    conv5 = BatchNormalization()(conv5)

    # right

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    up6 = Dropout(dropout_rate)(up6)
    conv6 = CommonConv2D(256, (3, 3))(up6)
    conv6 = CommonConv2D(256, (3, 3))(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    up7 = Dropout(dropout_rate)(up7)
    conv7 = CommonConv2D(128, (3, 3))(up7)
    conv7 = CommonConv2D(128, (3, 3))(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    up8 = Dropout(dropout_rate)(up8)
    conv8 = CommonConv2D(64, (3, 3))(up8)
    conv8 = CommonConv2D(64, (3, 3))(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    up9 = Dropout(dropout_rate)(up9)
    conv9 = CommonConv2D(32, (3, 3))(up9)
    conv9 = CommonConv2D(32, (3, 3))(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer=l1(1e-3), bias_regularizer=l1(1e-3), activity_regularizer=l1(1e-3))(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if load_weights:
        weight_path = os.path.join(model_weights_archive, "unet5.hdf5")
        if not os.path.exists(weight_path):
            import six.moves.urllib as urllib
            opener = urllib.request.URLopener()
            download_url = "https://github.com/korewayume/dapeng/releases/download/v1.0/release_v1.0.hdf5"
            opener.retrieve(download_url, weight_path)
        model.load_weights(weight_path)

    return model
