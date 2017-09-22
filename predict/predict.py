# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from dapeng.models.unet import unet5
from dapeng.utils.keras_tools import ImageDirectoryIterator, x_tif_reader, predict_from_directory

#configuration
input_path = "/root/input"
output_path = "/root/output"
dir_range = (108694, 108924)
filename_regex_unix_style = "*.jpeg"
weights_path = "ModelArchive/release_v1.0.hdf5"

model = unet5(256)
model.load_weights(weights_path)


def imsave(savepath, array):
    Image.fromarray(array, mode="L").save(savepath)


for row in tqdm(range(*dir_range), desc=u"预测"):
    glob_path = os.path.join(input_path, "{}/{}".format(row, filename_regex_unix_style))
    filepaths = glob(glob_path)
    gen = ImageDirectoryIterator([(filepath, None) for filepath in filepaths], x_tif_reader, None, batch_size=16,
                                 shuffle=False)
    predic_y = model.predict_generator(gen, steps=len(gen), verbose=0)
    predic_y_byte = (predic_y * 255).astype(np.uint8)
    for i, filepath in enumerate(filepaths):
        filename = os.path.basename(filepath)
        gray = predic_y_byte[i].reshape(256, 256)
        imsave(os.path.join(output_path, str(row) + "_" + os.path.splitext(filename)[0] + ".png"), gray)
