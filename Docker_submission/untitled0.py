# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:45:56 2022

@author: Administrateur
"""

### knights features

import pandas as pd
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import SimpleITK as sitk
import numpy as np
import os
import numpy as np
import skimage
#import skimage.io as io
import skimage.transform as transform
import torch

import numpy as np
import pandas as pd
import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import cv2
import matplotlib.pyplot as plt
import skimage 

#import logging
#import traceback
from typing import Optional, Tuple


def kits_normalization(input_image: np.ndarray):
    # first, clip to [-62, 310] (corresponds to 0.5 and 99.5 percentile in the foreground regions)
    # then, subtract 104.9 and divide by 75.3 (corresponds to mean and std in the foreground regions, respectively)
    clip_min = -62
    clip_max = 301
    mean_val = 104.0
    std_val = 75.3
    input_image = np.minimum(np.maximum(input_image, clip_min), clip_max)
    input_image -= mean_val
    input_image /= std_val
    return input_image

def normalize_to_range(input_image: np.ndarray, range: Tuple = (0.0, 1.0)):
    """
    Scales tensor to range
    @param input_image: image of shape (H x W x C)
    @param range:       bounds for normalization
    @return:            normalized image
    """
    max_val = input_image.max()
    min_val = input_image.min()
    if min_val == max_val == 0:
        return input_image
    input_image = input_image - min_val
    input_image = input_image / (max_val - min_val)
    input_image = input_image * (range[1] - range[0])
    input_image = input_image + range[0]
    return input_image

from typing import Iterable

def center_crop(np_image: np.ndarray,
                new_shape: Iterable[int],
                outside_val: float = 0
                ) -> np.ndarray:
    output_image = np.full(new_shape, outside_val, np_image.dtype)

    slices = tuple()
    offsets = tuple()
    for it, sh in enumerate(new_shape):
        size = sh // 2
        if it == 0:
            center = np_image.shape[it] - size
        else:
            center = (np_image.shape[it] // 2)
        start = center - size
        stop = center + size + (sh % 2)

        # computing what area of the original image will be in the cropped output
        slce = slice(max(0, start), min(np_image.shape[it], stop))
        slices += (slce,)

        # computing offset to pad if the crop is partly outside of the scan
        offset = slice(-min(0, start), 2 * size - max(0, (start + 2 * size) - np_image.shape[it]))
        offsets += (offset,)

    output_image[offsets] = np_image[slices]

    return output_image


def pad_image(image: np.ndarray, outer_height: int, outer_width: int, pad_value: Tuple):
    """
    Pastes input image in the middle of a larger one
    @param image:        image of shape (H x W x C)
    @param outer_height: final outer height
    @param outer_width:  final outer width
    @param pad_value:    value for padding around inner image
    @return:             padded image
    """
    inner_height, inner_width = image.shape[0], image.shape[1]
    h_offset = int((outer_height - inner_height) / 2.0)
    w_offset = int((outer_width - inner_width) / 2.0)
    outer_image = np.ones((outer_height, outer_width, 3), dtype=image.dtype) * pad_value
    outer_image[h_offset:h_offset + inner_height, w_offset:w_offset + inner_width, :] = image

    return outer_image

