import sys
sys.path.append(r'C:\Users\apidde\OneDrive - ICFO\Documents\Python Scripts\pyTFM-master')
sys.path.append(r'U:\Upright tracker\TRNs\newVersion\codes')
import helpers
from helpers.imageUtils import *
from pyTFM.TFM_functions import calculate_deformation
from pyTFM.plotting import show_quiver
import pyTFM.TFM_functions as TFM_functions
from pyTFM.TFM_functions import TFM_tractions
from pyTFM.TFM_functions import calculate_deformation
import random
import pyTFM.plotting as plotting

from openpiv import tools, pyprocess, validation, filters, scaling
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats as st
from scipy import signal
import pickle
import imageio
import importlib_resources
import pathlib
from tifffile import imread
import zarr
import os
import cv2
import skimage.exposure
import warnings
import glob
import pandas as pd
warnings.filterwarnings('ignore')
from scipy.ndimage import gaussian_filter, laplace

# parameters for image adjustment
gmax = 65535
# load image and load mask
morphKernelSize = 40
iterationsDilation = 8
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphKernelSize, morphKernelSize))
maxPadding = 100

#PIV params 
winsize = 42 # pixels, interrogation window size in frame A
searchsize = winsize + 1  # pixels, search area size in frame B
overlap = int(0.6 * winsize) # pixels, 50% overlap
thresh = 1.8 #1.5
fps = 8
dt = 1./fps
camSize = 6.5
binning = 2
magnification = 20
pixToUm = camSize * binning / magnification



def adjust(image, mini, maxi):
    image[image > maxi] = maxi
    image[image < mini] = mini
    image = (((image - mini) / (maxi - mini)) * 255).astype('uint8')
    return image
    
def load_adjust(fname, n1=0, n2=-1):
    lazytiff = imread(fname, aszarr = True)
    image0 = zarr.open(lazytiff, mode='r') 
    N, height, width = image0.shape
    if n2 == -1:
        n2 = N
    frames = range(n1, n2)
    A = np.zeros((len(frames), height, width), dtype='uint8') 
    for iframe, frame in enumerate(frames):
        image = adjust(image0[frame], 0, gmax)
        A[iframe] = image.astype('uint8')
    return A

def load_mask(fname, n1=0, n2=-1):
    mask = imread(fname)
    N, height, width = mask.shape
    if n2 == -1:
        n2 = N
    mask = mask[n1 : n2]   
    mask = (mask * 255).astype('uint8')
    return mask

def dilate_mask(mask, kernel, iterationsDilation):
    N, height, width = mask.shape
    mask_new = np.zeros((N, height, width), dtype=np.uint8)
    for i in range(N):
        mask0 = mask[i]
        dilation = cv2.dilate(mask0, kernel, iterations = iterationsDilation)  
        mask_new[i] = dilation
    return mask_new

def mask_where_nans(paddedA, x, y, u0, v0):
    u = u0.copy()
    v = v0.copy()
    idxs = np.isnan(paddedA)
    for ix in range(x.shape[1]):
        for iy in range(y.shape[0]):
             if np.isnan(paddedA[int(np.round(x[0, ix])), int(np.round(y[iy, 0]))]):
                 u[ix, iy] = np.nan
                 v[ix, iy] = np.nan
    return u, v
    
def display_offset(a, b, paddedA, shiftedB):
    height, width = a.shape
    orig = np.zeros((height, width, 3), dtype = 'uint8')
    orig[:, :, 0] = a
    orig[:, :, 1] = b        
    test = np.zeros((height, width, 3), dtype = 'uint8')
    print(paddedA.shape, shiftedB.shape)
    test[:, :, 0] = paddedA
    test[: ,:, 1] = shiftedB
    
    fig, axs = plt.subplots(ncols=2, figsize = (10, 4), sharex=True, sharey=True)
    axs[0].imshow(orig)
    axs[0].set_title('original')
    axs[1].imshow(test)
    axs[1].set_title('corrected')
    axs[0].set_yticks([])
    axs[0].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xticks([])
    fig.tight_layout()
    #fig.savefig('offsetCorrection.png')
    return fig, axs

def smooth_mask(mask):
    N, height, width = mask.shape
    mask_new = np.zeros((N, height, width), dtype=np.uint8)
    for i in range(N):
        mask0 = mask[i]
        blur = cv2.GaussianBlur(mask0, (0,0), sigmaX=9, sigmaY=9, borderType = cv2.BORDER_DEFAULT)
        result = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))
        result = result.astype('uint8')
        retval, maskt = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        mask_new[i] = maskt
    return mask_new

def process_mask(fname, kernel, iterationsDilation, n1=0, n2=-1):
    mask = load_mask(fname, n1=n1, n2=n2)
    mask = smooth_mask(mask)
    dilated = dilate_mask(mask, kernel, iterationsDilation) 
    return mask, dilated


def find_shift(ima, imb, padding_size = None, crop=True, black_out=True, use_laplace=True):
    
    size = ima.shape
    corr = pyprocess.fft_correlate_images(ima, imb)
    if not use_laplace:
        mtx = corr
    else:
        mtx = np.abs(laplace(corr))
    ind = np.unravel_index(np.argmax(mtx, axis=None), corr.shape)
    shift = np.array(corr.shape) // 2 - ind
    if padding_size is None:
        padding_size = np.max(abs(shift))
    paddedA = np.zeros((size[0] + 2 * padding_size, size[1] + 2 * padding_size), dtype = 'uint8')
    paddedA[padding_size : padding_size + size[0], padding_size : padding_size + size[1]] = ima
    shiftedB = np.zeros((size[0] + 2 * padding_size, size[1] + 2 * padding_size), dtype = 'uint8')
    shiftedB[padding_size + shift[0] : padding_size + shift[0] + size[0], padding_size + shift[1] : padding_size + shift[1] + size[1]] = imb
    if crop:
        paddedA = paddedA[padding_size:-padding_size, padding_size:-padding_size]
        shiftedB = shiftedB[padding_size:-padding_size, padding_size:-padding_size]
    if black_out:
        if shift[0] > 0:
            paddedA[0:shift[0]] = 0
        elif shift[0] < 0:
            paddedA[shift[0]:] = 0
        if shift[1] > 0:
            paddedA[:, 0:shift[1]] = 0
        elif shift[1] < 0:
            paddedA[:, shift[1]:] = 0
    return shift, paddedA, shiftedB

def mask_nans_images(a, b):
    idxs = np.isnan(a) | np.isnan(b)
    a[idxs] = 0
    b[idxs] = 0
    a = a.astype('uint8')
    b = b.astype('uint8')
    return a, b
    
def find_shift_frames(frames, maxPadding=maxPadding, display=False, crop=True, black_out=True):
    N, height, width = frames.shape
    Offset = np.zeros((N, 2), dtype='int32')
    for i in range(N - 1):
        a = frames[i]
        b = frames[i + 1]
        a, b = mask_nans_images(a, b)
        offset, paddedA, shiftedB = find_shift(a, b, padding_size=None, crop=crop, black_out=black_out, use_laplace=True)
        Offset[i] = offset
        if display:
            try:
                print(offset)
                orig = np.zeros((height, width, 3), dtype = 'uint8')
                orig[:, :, 1] = a
                orig[:, :, 2] = b        
                test = np.zeros((height, width, 3), dtype = 'uint8')
                print(paddedA.shape, shiftedB.shape)
                test[:, :, 1] = paddedA
                test[: ,:, 2] = shiftedB
                image = cv2.putText(test, str(offset), org, font,  
                               fontScale, color, thickness, cv2.LINE_AA) 
                cv2.imshow('orig', orig)
                cv2.imshow('image', image)
                k = cv2.waitKey(10)
                if k == ord('q'):
                    break
            except:
                pass
    if display:
        cv2.destroyAllWindows()
    return Offset
 