from helper.process_tractions import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy import signal
import pickle
import imageio
from tifffile import imread
import zarr
import os
import cv2
import warnings
import glob

n_profile_points = 100
n_body_points = 100

coordinates = np.linspace(0, 1, n_body_points)

fnameCSV = 'skeleton_file_path_skel.csv'
skel_array = np.loadtxt(fnameCSV).reshape(-1, n_body_points, 2)
N = skel_array.shape[0]
T = N / fps
t = np.linspace(0, T, N)
    
# read tractions 
dict_file = 'traction_dictionary_file.pkl'
with open(dict_file, 'rb') as fp:
    data = pickle.load(fp)
selected = data['selected']

Tx = data['tx']
Ty = data['ty']
traction_array0 = np.sqrt(Tx**2 + Ty**2)
traction_array = expand_image(traction_array0, data['worm'].shape[1])
traction_array_x = expand_image(Tx, data['worm'].shape[1])
traction_array_y = expand_image(Ty, data['worm'].shape[1])

# find curvature
curv = curvature_array(skel_array)
curv_selected = curv[selected]


# find profile
profiles_array = extract_profiles_array(traction_array, skel_array[selected], n_profile_points)
profiles_array_n, profiles_array_t = extract_profiles_nt(traction_array_x, traction_array_y, skel_array[selected], n_profile_points)

N = profiles_array.shape[0]
arrays = [profiles_array, profiles_array_n, profiles_array_t]

X = skel_array[:, :, 0]
Y = skel_array[:, :, 1]
Theta = calculateCurvature(X, Y, len(coordinates), s=0.1) / np.pi

fig, axs = plt.subplots(nrows=len(arrays) + 1, sharex=True, sharey=True, figsize=(15, 8))
amp = 1
weight = 0.02

# smoothing, denoising, can be skipped 
theta = preprocess(Theta.T, weight=weight)

pc = axs[0].pcolor(t, coordinates, theta, cmap='RdBu', vmin=-amp, vmax=amp)
plt.colorbar(pc, label = '$\Theta$')
axs[0].spines[['right', 'top']].set_visible(False)
for ip, array in enumerate(arrays):
    # body point 
    maxtraction = get_abs_percentile(array, 97.5, axis=2) 
    
    # smoothing, denoising, can be skipped 
    arr = preprocess(maxtraction.T, weight=weight)
    
    vmax = st.scoreatpercentile(abs(maxtraction.flatten()), 95)
    if ip == 0:
        vmin = 0
    else:
        vmin = -vmax
    pc = axs[ip + 1].pcolor(t, coordinates, arr, cmap='YlGnBu', vmin=vmin, vmax=vmax)
    axs[ip + 1].set_title(labels[ip])
    if ip == 1:
        plt.colorbar(pc, label='traction (Pa)')
        axs[ip + 1].set_ylabel('Body coordinate')
    else:
        plt.colorbar(pc)
    axs[ip + 1].spines[['right', 'top']].set_visible(False)
axs[0].set_title(subfolder)
axs[-1].set_xlabel('Time (sec)')

