# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4, 2023

@author: apidde
"""

import tifffile as tif
import scipy.stats as st
import numpy as np
#print(sys.version)
import imutils
from helpers.trackCells import *
from helpers.skeletonise import *

def load(imfile):
    return tif.imread(imfile)

def resize(image, binning):
    shape = list(image.shape)
    N = shape[0]
    image = image.reshape(-1, shape[-2], shape[-1])
    shape[-2] = shape[-2] // binning
    shape[-1] = WIDTH = shape[-1] // binning
    print(shape, WIDTH)
    newImage = np.zeros(shape = shape, dtype=image.dtype)
    newImage = newImage.reshape(-1, shape[-2], shape[-1])
    for i in range(newImage.shape[0]):
        frame = image[i]
        frame = imutils.resize(frame, width = WIDTH)
        newImage[i] = frame
    newImage = newImage.reshape(shape)
    return newImage

def adjust_contrast(image, percmin, percmax, fromPerc=True):
    N, height, width = image.shape
    if fromPerc:
        vmin = st.scoreatpercentile(image, percmin)
        vmax = st.scoreatpercentile(image, percmax)
    else:
        vmin = percmin
        vmax = percmax
    print('val min ', vmin, ' , vmax ', vmax)
    newImage = np.zeros(shape = image.shape, dtype = np.uint8)
    for i in range(N):
        frame = image[i]
        frame[frame >= vmax] = vmax
        frame[frame <= vmin] = vmin
        frame = ((frame - vmin) / vmax * 255).astype("uint8")
        newImage[i] = frame
    return newImage


