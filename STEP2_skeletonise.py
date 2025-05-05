import sys
import os
from tifffile import imread, imwrite
import scipy.stats as st
import cv2
import sys
import numpy as np
#print(sys.version)
import imutils
import matplotlib.pyplot as plt
import glob 
import zarr
import pickle
from helpers.load_preprocess import * 
from helpers.separate_worm_bckg import *
from helpers.skeletonise import *  

binning_additional = 2
nskel_points = 100

desired_length = 900 // binning_additional
deviation = 300 // binning_additional

file_path = 'path_to_tif_file'
file_name = 'file_name.tif'
file_name = os.path.split(im_file)
file_name0 = file_name.rsplit('.', 1)[0]

save_path = 'path_to_save_the_skeleton'
dict_file = 'full_path_to_dictionary_returned_in_previous_step.pkl'
            
fnameCSV = os.path.join(save_path0, 'skel.csv')
with open(dict_file, 'rb') as fp:
    print('loading dictionary')
    data = pickle.load(fp)
    
selected = data['selected']
worm = data['worm']
worm_resized = resize(worm, binning_additional)
SkelArray0, S, IsSkelGood, Frame = skeletonise_array(worm_resized, nskel_points, desired_length=desired_length, deviation=deviation, image=worm_resized)
SkelArray = SkelArray0 * binning_additional
skel_reshaped = SkelArray.reshape(SkelArray.shape[0], -1)

if os.path.isfile(fnameCSV):
    os.remove(fnameCSV)
np.savetxt(fnameCSV, skel_reshaped)



