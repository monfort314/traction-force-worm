import sys
from helpers.image_utils import *
from helpers.find_offset import *
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
warnings.filterwarnings('ignore')

# load image and load mask
morph_kernel_size = 40
iterations_dilation = 8
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
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



    
def find_PIV(frames, Shift, display_offsetting=False):
    kernel = np.ones((3, 3)) / 9
    N, height, width = frames.shape
    nrows, ncols = pyprocess.get_field_shape([height, width], searchsize, overlap)
    U = np.zeros((N, nrows, ncols))
    V = np.zeros((N, nrows, ncols))
    for i in range(N - 1):
        a = frames[i]
        b = frames[i + 1]
        paddedA0, shiftedB0 = shiftImage(Shift[i], a, b, padding_size = None, crop=True, black_out=True)
        paddedA, shiftedB = mask_nans_images(paddedA0, shiftedB0)
        u0, v0, sig2noise = pyprocess.extended_search_area_piv(
            paddedA.astype(np.int32),
            shiftedB.astype(np.int32),
            window_size=winsize,
            overlap=overlap,
            #dt=dt,
            search_area_size=searchsize,
            sig2noise_method='peak2peak',
        )
        x, y = pyprocess.get_coordinates(
            image_size=paddedA.shape,
            search_area_size=searchsize,
            overlap=overlap)
        
        u2, v2 = mask_where_nans(a, x, y, u0, v0)
        x, y, u3, v3 = scaling.uniform(
            x, y, u2, v2,
            scaling_factor = 1#1e3 / pixToUm,  # 96.52 pixels/millimeter
        )
        # 0,0 shall be bottom left, positive rotation rate is counterclockwise
        x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)
        # mask the lowest and highest percentiles
        U[i] = u3
        V[i] = v3
    if display_offsetting:                
        cv2.destroyAllWindows()
    return x, y, U, V


def save_data(Worm, Background, u, v, tx, ty, fname, save_path, selected):
    data = {'worm': Worm,
           'background': Background,
           'u': u, 'v': v, 'tx': tx, 'ty': ty, 'selected': selected}
    saveFile = os.path.join(save_path, fname + '.pkl')
    print(saveFile)
    print("Approximate size of data:", sys.getsizeof(data) // (1024**2), "MB")
    try:
        with open(saveFile, 'wb') as fp:
            pickle.dump(data, fp)
        return 0
    except OSError as e:
        print(f"Error saving file: {e}")
        return [data, saveFile]


# now finally find particle image velocimetry
import scipy.stats as st
from scipy import signal
import pickle

def display_vector_field(
    im, x, y, u, v, flags, mask,
    on_img=False,
    image_name=None,
    window_size=32,
    scaling_factor=1.,
    ax=None,
    width=0.0025,
    show_invalid=True
):
    """ Displays quiver plot of the data 
    
    Parameters
    ----------

    on_img : Bool, optional
        if True, display the vector field on top of the image provided by 
        image_name

    image_name : string, optional
        path to the image to plot the vector field onto when on_img is True

    window_size : int, optional
        when on_img is True, provide the interrogation window size to fit the 
        background image to the vector field

    scaling_factor : float, optional
        when on_img is True, provide the scaling factor to scale the background
        image to the vector field
    
    show_invalid: bool, show or not the invalid vectors, default is True

        
    Key arguments   : (additional parameters, optional)
        *scale*: [None | float]
        *width*: [None | float]
    
    
    See also:
    ---------
    matplotlib.pyplot.quiver
    
        
    Examples
    --------
    --- only vector field
    >>> openpiv.tools.display_vector_field('./exp1_0000.txt',scale=100, 
                                           width=0.0025) 

    --- vector field on top of image
    >>> openpiv.tools.display_vector_field(Path('./exp1_0000.txt'), on_img=True, 
                                          image_name=Path('exp1_001_a.bmp'), 
                                          window_size=32, scaling_factor=70, 
                                          scale=100, width=0.0025)
    
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if on_img is True:  # plot a background image
        #im = negative(im)  # plot negative of the image for more clarity
        xmax = np.amax(x) + window_size / (2 * scaling_factor)
        ymax = np.amax(y) + window_size / (2 * scaling_factor)
        ax.imshow(im, cmap="Greys_r", extent=[0.0, xmax, 0.0, ymax])  
        #ax.imshow(im, cmap="Greys_r") 


    # first mask whatever has to be masked
    u[mask.astype(bool)] = 0.
    v[mask.astype(bool)] = 0.
    
    # now mark the valid/invalid vectors
    invalid = flags > 0 # mask.astype("bool")  
    valid = ~invalid
    ax.quiver(
        x[valid],
        y[valid],
        u[valid],
        v[valid],
        color="c",
        width=width,
        )
        
    if show_invalid and len(invalid) > 0:
        ax.quiver(
                x[invalid],
                y[invalid],
                u[invalid],
                v[invalid],
                color="r",
                width=width,
                )
    ax.set_aspect(1.)
    plt.show()

    return fig, ax

def display_vector_field(frames, x, y, U, V, fname, ax = None, fig=None, save=True):
    N, height, width = frames.shape
    if ax is None or fig is None:
        fig, ax = plt.subplots()
    plt.ion()
    def plotFrame(i):
        frame = frames[i]
        mask = np.zeros(x.shape)
        ax.clear()
        display_vector_field(frame, x, y, U[i], V[i], mask, mask, scaling_factor = 1e3 / pixToUm,
            width=0.01, # width is the thickness of the arrow
            on_img=True, # overlay on the image
            ax = ax, 
            window_size=winsize
        )
        ax.set_yticks([])
        ax.set_xticks([])
    anim = animation.FuncAnimation(fig, plotFrame, frames=range(0, len(frames)), interval=125, blit=False, repeat=False)
    if save:
        print('saving animation')
        writervideo = animation.ImageMagickWriter(fps=4)
        fnameSave2 = fname + '_animation.gif'
        anim.save(fnameSave2, writer = writervideo)
    
    
def main(file_name, file_path, mask_file, save_path, stiffness):
    print('analysing ', file_name)
    filename = os.path.join(file_path, file_name)
    filename0 = file_name[:-4]
    frames = load_adjust(filename)
    
    worm, dilated = process_mask(mask_file, kernel, iterations_dilation, n1=0, n2=-1)

    background = np.nan * np.zeros(frames.shape, dtype='float')
    background[dilated == 0] = frames[dilated == 0]
    surrounding = np.nan * np.zeros(frames.shape, dtype='float')
    idxs = (dilated == 255) & (worm == 0)
    surrounding[idxs] = frames[idxs]
    print('offset')
    offset = find_shift_frames(background)
    print('PIV')
    x, y, U, V = find_PIV(surrounding, offset)
    print('animation')
    display_vector_field(frames, x, y, U, V, filename[:-4], save=True)
    
    N, height, width = frames.shape
    ps1 = pixToUm # pixel size of the image of the beads
    im1_shape = (height, width) # dimensions of the image of the beads
    ps2 = ps1 * np.mean(np.array(im1_shape) / np.array(x.shape)) # pixel size of of the deformation field
    young = stiffness * 1000 # Young's modulus of the substrate in Pa
    sigma = 0.457 # Poisson's ratio of the substrate
    h = 1000 # height of the substrate in µm, "infinite" is also accepted
    cmap = 'Oranges'
    filter_size = 18
    selected = range(N)
    print('estimating only for all')
    Tx, Ty = np.zeros((len(selected), x.shape[0], x.shape[1])), np.zeros((len(selected), x.shape[0], x.shape[1]))
    print('tractions')
    for k in range(len(selected)):
        i = selected[k]
        frame = frames[i,:, :]
        mask = np.zeros(x.shape) 
        u = U[i].copy()
        v = -V[i].copy()
        u[np.isnan(u)] = 0
        v[np.isnan(v)] = 0
        
        try:
            tx, ty = TFM_tractions(u, v, pixelsize1=ps1, pixelsize2=ps2, h=h, young=young, sigma=sigma, fs=filter_size)
            tx[np.isnan(U[i])] = 0
            ty[np.isnan(V[i])] = 0
        except Exception as error:
            tx, ty = np.zeros((u.shape[0], u.shape[1])), np.zeros((u.shape[0], u.shape[1]))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, file_name, exc_tb.tb_lineno)
            print('could not estimate traction ', i)
            pass
        
        Tx[k] = tx
        Ty[k] = ty
        
    return save_data(worm, background, U, V, Tx, Ty, filename0, save_path, selected)


if __name__ == "__main__":
    
    stiffness = '' # in kPa
    file_path = '' # path to image file
    file_name = '' #
    save_path = '' # path where to save the results

    file_name0 = file_name[:-4]
    
    mask_file = 'path_to_segmented_image_of_worm' # os.path.join(save_path,  'committed_objects.tif')
    
    output = main(file_name, file_path, mask_file, save_path, stiffness)

