import helpers
from helpers.image_utils import *
from helpers.skeletonise import *
from pyTFM.TFM_functions import calculate_deformation

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
warnings.filterwarnings('ignore')

import imutils
from scipy.ndimage import map_coordinates
import seaborn as sns
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import gaussian_filter

def interpolate_nan_image_gaussian(image):
    nan_mask = np.isnan(image)
    image_no_nan = np.nan_to_num(image, nan=0)  # Replace NaN with 0 temporarily
    smoothed = gaussian_filter(image_no_nan, sigma=3)
    image_filled = np.where(nan_mask, smoothed, image)
    return image_filled

def preprocess(image, weight=0.7, perc = [1, 99]):
    #perc_min = perc[0]
    #perc_max = perc[1]
    #image[(image < st.scoreatpercentile(image, perc_min)) | (image > st.scoreatpercentile(image, perc_max))] = np.nan
    mini = np.nanmin(image)
    maxi = np.nanmax(image)
    int_gaussian = interpolate_nan_image_gaussian(image)
    image = int_gaussian
    
    image = (image - mini) / (maxi - mini)
    img = denoise_tv_chambolle(image, weight=weight)
    img = img * (maxi - mini) + mini
    return img 

def extract_perpendicular_profiles(image, skel, r):
    # Array to store the pixel values for each skeleton point
    profiles = []
    for i in range(len(skel)):
        # For each point, get the direction to the next point to define the tangent
        if i < len(skel) - 1:
            dx, dy = skel[i + 1] - skel[i]
        else:
            dx, dy = skel[i] - skel[i - 1]
        
        # Calculate the perpendicular direction (-dy, dx)
        perp_dir = np.array([dy, -dx])
        perp_dir = perp_dir / np.linalg.norm(perp_dir)  # Normalize the perpendicular vector
        
        # Define a set of points along the perpendicular line centered at the skeleton point
        distances = np.linspace(-r, r, 2*r)
        perpendicular_points = skel[i] + np.outer(distances, perp_dir)
        
        # Extract the pixel values using interpolation
        pixel_vals = map_coordinates(image, [perpendicular_points[:, 1], perpendicular_points[:, 0]], order=1, mode='nearest')
        
        profiles.append(pixel_vals)
    
    # Convert the list to a numpy array (50 x 200)
    profiles = np.array(profiles)
    return profiles

def extract_perpendicular_profiles_nt(Tx, Ty, skel, r):
    # Array to store the pixel values for each skeleton point
    profiles_n = []
    profiles_t = []
    
    for i in range(len(skel)):
        # For each point, get the direction to the next point to define the tangent
        if i < len(skel) - 1:
            dx, dy = skel[i + 1] - skel[i]
        else:
            dx, dy = skel[i] - skel[i - 1]
            
        tang_dir = np.array([dx, dy])
        tang_dir = tang_dir / np.linalg.norm(tang_dir)  # Normalize the tangential vector
        
        # Calculate the perpendicular direction (-dy, dx)
        perp_dir = np.array([dy, -dx])
        perp_dir = perp_dir / np.linalg.norm(perp_dir)  # Normalize the perpendicular vector
        
        # Define a set of points along the perpendicular line centered at the skeleton point
        distances = np.linspace(-r, r, 2*r)
        perpendicular_points = skel[i] + np.outer(distances, perp_dir)
        
        # Extract the pixel values using interpolation
        Tx_vals = map_coordinates(Tx, [perpendicular_points[:, 1], perpendicular_points[:, 0]], order=1, mode='nearest')
        Ty_vals = map_coordinates(Ty, [perpendicular_points[:, 1], perpendicular_points[:, 0]], order=1, mode='nearest')
        T_profile = np.vstack((Tx_vals, Ty_vals))
        #print(T_profile.shape)
        T_n = np.dot(T_profile.T, perp_dir)
        T_t = np.dot(T_profile.T, tang_dir)
        
        profiles_n.append(T_n)
        profiles_t.append(T_t)
    
    # Convert the list to a numpy array (50 x 200)
    profiles_n = np.array(profiles_n)
    profiles_t = np.array(profiles_t)
    return profiles_n, profiles_t
    
def extract_profiles_array(image_array, skel_array, n_profile_points):
    N, n_body_points, _ = skel_array.shape
    profiles_array = np.zeros((N, n_body_points, 2 * n_profile_points))
    for i in range(N):
        profile = extract_perpendicular_profiles(image_array[i], skel_array[i], n_profile_points)
        profiles_array[i] = profile
    return profiles_array

def extract_profiles_nt(image_array_x, image_array_y, skel_array, n_profile_points):
    N, n_body_points, _ = skel_array.shape
    profiles_array_n = np.zeros((N, n_body_points, 2 * n_profile_points))
    profiles_array_t = np.zeros((N, n_body_points, 2 * n_profile_points))
    for i in range(N):
        profile_n, profile_t = extract_perpendicular_profiles_nt(image_array_x[i], image_array_y[i], skel_array[i], n_profile_points)
        profiles_array_n[i] = profile_n
        profiles_array_t[i] = profile_t
    return profiles_array_n, profiles_array_t
    
def curvature_array(skel_array):
    N, nbody_points, _ = skel_array.shape
    curvature_array = np.zeros((N, nbody_points))
    for i in range(N):
        curvature_array[i] = curvature(skel_array[i, :, 0], skel_array[i, :, 1])
    return curvature_array
    
def curvature(x, y):
    # Compute first derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Compute second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Compute curvature
    curvature = np.abs(dx * ddy - dy * ddx)/ (dx**2 + dy**2)**(3/2)
    return curvature

def expand_image(image_array, new_width):
    N = image_array.shape[0]
    new_image_array = np.zeros((N, new_width, new_width), dtype=image_array.dtype)
    for i in range(N):
        new_image_array[i] = imutils.resize(image_array[i], width = new_width)
    return new_image_array

def get_abs_percentile(A, p, axis):
    """
    Calculate the percentile of the absolute values of A along the specified axis,
    while preserving the sign of the original values closest to the percentile value.
    
    Parameters:
    A (ndarray): Input array.
    p (float): Percentile to compute (0-100).
    axis (int): Axis along which to compute the percentiles.
    
    Returns:
    ndarray: Array of the percentile values with the original signs preserved,
             with reduced dimension along the specified axis.
    """
    # Compute the absolute values
    abs_A = np.abs(A)
    
    # Compute the specified percentile along the given axis
    abs_percentile = np.percentile(abs_A, p, axis=axis, keepdims=True)
    
    # Find the index of the closest value to the computed percentile in abs_A
    abs_diff = np.abs(abs_A - abs_percentile)
    closest_indices = np.argmin(abs_diff, axis=axis, keepdims=True)
    
    # Gather the signs of the values closest to the percentile
    # Use advanced indexing to select the sign of the closest values
    result_signs = np.take_along_axis(np.sign(A), closest_indices, axis=axis)
    
    # Multiply the percentile result by the sign of the closest values
    result = result_signs * abs_percentile
    
    return np.squeeze(result)
    
