# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
import re, json, os
from datetime import datetime
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from dapeng.models.unet import unet5
from dapeng.metrics import jaccard_coefficient
from dapeng.utils.keras_tools import x_tif_reader, y_tif_reader, train_from_directories, random_transform

with open("u-net-train.json") as cfg:
    configuration = json.load(cfg)
unet_model_path = configuration.get("unet_model_path", "ModelArchive/u-net-model.yaml")
unet_weight_path = configuration.get("unet_weight_path", 'ModelArchive/unet_best_checkpoint.hdf5')
unet_weight_checkpoint_path = configuration.get("unet_weight_checkpoint_path", 'ModelArchive/unet_best_checkpoint.hdf5')
tensorboard_log_directory = os.path.join(configuration["tensorboard_log_directory"], datetime.now().strftime("%m-%d-%H-%M"))
load_weights = configuration.get("load_weights", False)
fit_verbose = configuration.get("fit_verbose", True)
mosaic_size = configuration.get("mosaic_size", 256)
learning_rate = configuration.get("learning_rate", 1e-3)
dropout_rate = configuration.get("dropout_rate", 0.2)
epochs = configuration.get("epochs", 300)
batch_size = configuration.get("batch_size", 15)
train_x_path = configuration["train_x_path"]
train_y_path = configuration["train_y_path"]
valid_x_path = configuration["valid_x_path"]
valid_y_path = configuration["valid_y_path"]
valid_batch_size = configuration.get("valid_batch_size", 10)
random_rotate_angle = configuration.get("random_rotate_angle", 15)
random_rotate_mode = configuration.get("random_rotate_mode", "reflect")


model = unet5(
    mosaic_size,
    optimizer=Adam(lr=learning_rate),
    loss='binary_crossentropy',
    metrics=[jaccard_coefficient],
    load_weights=load_weights
)

model_checkpoint = ModelCheckpoint(unet_weight_checkpoint_path, monitor='loss', save_best_only=True, verbose=0)
tensorboard = TensorBoard(log_dir=tensorboard_log_directory, write_graph=True, write_images=True)
model_callbacks = [model_checkpoint, tensorboard]


filename_regex = re.compile(r"^\d{3}_\d{3}_\d{3}\.tif")
print("正在扫描训练样本")
train_gen = train_from_directories(train_x_path, train_y_path, x_tif_reader, y_tif_reader,
                                   filename_regex, batch_size=batch_size,
                                   random_transform_func=random_transform(random_rotate_angle, mode=random_rotate_mode))
print("找到训练样本：{}".format(len(train_gen)))
print("正在扫描验证样本")
valid_gen = train_from_directories(valid_x_path, valid_y_path, x_tif_reader, y_tif_reader,
                                   filename_regex, batch_size=valid_batch_size)
print("找到验证样本：{}".format(len(valid_gen)))


print("开始训练网络")
model.fit_generator(train_gen, steps_per_epoch=len(train_gen),
                    validation_data=valid_gen, validation_steps=len(valid_gen),
                    epochs=epochs, verbose=fit_verbose, callbacks=model_callbacks, shuffle=True)
