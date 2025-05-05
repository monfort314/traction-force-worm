from helpers.image_utils import *
from helpers.skeletonise import *
import random
import numpy as np
import matplotlib.pyplot as plt
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
warnings.filterwarnings('ignore')



fnameCSV = 'path_to_skeleton_skel.csv'
image_file = 'path_to_image.tif'

n_profile_points = 100
n_body_points = 100
fps = 8

coordinates = np.linspace(0, 1, n_body_points)
skel_array = np.loadtxt(fnameCSV).reshape(-1, n_body_points, 2)
N = skel_array.shape[0]
T = N / fps
t = np.linspace(0, T, N)

X = skel_array[:, :, 0]
Y = skel_array[:, :, 1]
Theta = calculateCurvature(X, Y, len(coordinates), s=0.1) / np.pi

images = zarr.open(imread(image_file, aszarr = True))
image = images[0]
            
idxs = random.sample(range(1, X.shape[0]), 3) 

fig, axs = plt.subplots(nrows=2, ncols=2)
amp = 1 
pc = axs[0, 0].pcolor(t, coordinates, Theta.T, cmap='RdBu', vmin=-amp, vmax=amp)
for i, idx in enumerate(idxs):
    image = images[idx]
    ax = axs[(i + 1) // 2, (i + 1) % 2]
    ax.imshow(image, cmap='Greys_r', vmin = 0, vmax = 0.7 * image.max())
    ax.plot(X[idx], Y[idx])
    ax.plot(X[idx, 0], Y[idx, 0], 'rx')
    ax.set_title(idx)
plt.show()
