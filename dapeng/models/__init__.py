# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
from . import unet

model_weights_archive = "~/.dapeng/weights/"

if not os.path.exists(model_weights_archive):
    os.makedirs(model_weights_archive)
