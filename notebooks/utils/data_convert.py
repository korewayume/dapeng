# -*- coding: utf-8 -*-
import os, shutil, random, re
from glob import glob
from keras_tools import y_tif_reader
import numpy as np


# from_x_path = "/Users/mac/Downloads/重新做的北镇样本数据和原始数据/Workspace_W100_H100_F1-1_TO100-5_2017_09_13_1014_raw"
# from_y_path = "/Users/mac/Downloads/重新做的北镇样本数据和原始数据/Workspace_W100_H100_F1-1_TO100-5_2017_09_13_1014"
# to_train_x_path = "/Users/mac/Downloads/data/train/x"
# to_train_y_path = "/Users/mac/Downloads/data/train/y"
# to_validation_x_path = "/Users/mac/Downloads/data/validation/x"
# to_validation_y_path = "/Users/mac/Downloads/data/validation/y"


def newname(name):
    regex = re.compile(r"cell_(\d+)_(\d+)\.tif")
    match = re.match(regex, name)
    if match:
        return "{:03}_{:03}.tif".format(int(match.group(1)), int(match.group(2)))
    else:
        return name


def merge_dir(batch_no, from_x_directory, from_y_directory, to_x_directory, to_y_directory):
    filenames = os.listdir(from_y_directory)
    for filename in filenames:
        try:
            os.rename(os.path.join(from_x_directory, filename),
                      os.path.join(to_x_directory, "{:03}_{}".format(batch_no, newname(filename))))
            os.rename(os.path.join(from_y_directory, filename),
                      os.path.join(to_y_directory, "{:03}_{}".format(batch_no, newname(filename))))
        except OSError:
            print batch_no, filename


def train_test_split(batch_no, from_x_directory, from_y_directory, to_train_x_directory, to_train_y_directory,
                     to_test_x_directory, to_test_y_directory):
    image_paths = {path for path in glob(os.path.join(from_y_directory, "{:03}_*.tif".format(batch_no)))}
    has_dapeng = {path for path in image_paths if not np.all(y_tif_reader(path) == 0)}
    not_dapeng = image_paths - has_dapeng
    choice_n = len(has_dapeng) // 8
    has_dapeng = list(has_dapeng)
    not_dapeng = list(not_dapeng)
    has_dapeng_test = np.random.choice(np.array(has_dapeng), choice_n, replace=False).tolist()
    not_dapeng_test = np.random.choice(np.array(not_dapeng), choice_n, replace=False).tolist()
    to_train_y = image_paths - set(has_dapeng_test + not_dapeng_test)
    for from_y in has_dapeng_test + not_dapeng_test:
        filename = os.path.basename(from_y)
        shutil.move(os.path.join(from_x_directory, filename), os.path.join(to_test_x_directory, filename))
        shutil.move(os.path.join(from_y_directory, filename), os.path.join(to_test_y_directory, filename))
    # for from_y in to_train_y:
    #     filename = os.path.basename(from_y)
    #     shutil.move(os.path.join(from_x_directory, filename), os.path.join(to_train_x_directory, filename))
    #     shutil.move(os.path.join(from_y_directory, filename), os.path.join(to_train_y_directory, filename))


if __name__ == '__main__':
    xpaths = [
        "/Users/mac/Desktop/beizhen/重新做的北镇样本数据和原始数据/Workspace_W100_H100_F1-1_TO100-5_2017_09_13_1014_raw",
        "/Users/mac/Desktop/beizhen/北镇第二批数据/Workspace_W100_H100_F16-1_TO25-5_2017_09_13_raw",
        "/Users/mac/Desktop/beizhen/北镇第三批数据/Workspace_W100_H100_F1-11_TO15-15_2017_09_14_1346_raw",
        "/Users/mac/Desktop/beizhen/北镇第四批数据/Workspace_W100_H100_F21-11_TO25-15_2017_09_14_raw",
        "/Users/mac/Desktop/beizhen/北镇第五批数据/Workspace_W100_H100_F31-11_TO35-15_2017_09_14_raw"

    ]
    ypaths = [
        "/Users/mac/Desktop/beizhen/重新做的北镇样本数据和原始数据/Workspace_W100_H100_F1-1_TO100-5_2017_09_13_1014",
        "/Users/mac/Desktop/beizhen/北镇第二批数据/Workspace_W100_H100_F16-1_TO25-5_2017_09_13",
        "/Users/mac/Desktop/beizhen/北镇第三批数据/Workspace_W100_H100_F1-11_TO15-15_2017_09_14_1346",
        "/Users/mac/Desktop/beizhen/北镇第四批数据/Workspace_W100_H100_F21-11_TO25-15_2017_09_14",
        "/Users/mac/Desktop/beizhen/北镇第五批数据/Workspace_W100_H100_F31-11_TO35-15_2017_09_14"
    ]
    for batch, x_path, y_path in zip([1, 2, 3, 4, 5], xpaths, ypaths):
        merge_dir(batch, x_path, y_path, "/Users/mac/Desktop/beizhen/data/train/x", "/Users/mac/Desktop/beizhen/data/train/y")

    for i in [1, 2, 3, 4, 5]:
        train_test_split(i, "/Users/mac/Desktop/beizhen/data/train/x", "/Users/mac/Desktop/beizhen/data/train/y",
                         "/Users/mac/Desktop/beizhen/data/train/x", "/Users/mac/Desktop/beizhen/data/train/y",
                         "/Users/mac/Desktop/beizhen/data/validation/x", "/Users/mac/Desktop/beizhen/data/validation/y")
