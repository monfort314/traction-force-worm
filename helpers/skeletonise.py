# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4, 2023

@author: apidde
"""
import os
import cv2
import numpy as np
from scipy import interpolate
from fil_finder import FilFinder2D
import astropy.units as u
import matplotlib.pyplot as plt
import skimage.exposure
import matplotlib.animation as animation
from IPython import display
import pandas as pd
import scipy.io as sio
from scipy import stats
from scipy.interpolate import splev, splrep
from helpers.load_process_time_series import *
from numba import jit, prange
import pyvista as pv
import seaborn as sns


@jit(nopython=True, parallel=True)
def compute_density(values, positions):
    n_positions = positions.shape[1]  # Number of grid points
    n_values = values.shape[1]        # Number of data points
    density = np.zeros(n_positions)
    
    for i in prange(n_positions):
        density_i = 0.0
        for j in range(n_values):
            diff = values[:, j] - positions[:, i]
            diff_sq = np.dot(diff, diff)
            density_i += np.exp(-0.5 * diff_sq)
        density[i] = density_i
    
    density /= (2 * np.pi) ** (values.shape[0] / 2)  # Normalize density
    density /= n_values  # Normalize by number of data points
    return density


def find_skeleton(imageBinary, startpos, endpos):
    maxSteps = 10000
    image = imageBinary.copy()
    currentpos = startpos
    NoSteps = 0
    resultX = []
    resultY = []
    height, width = imageBinary.shape
    while (not (currentpos == endpos).all()) and (NoSteps < maxSteps):
        # black out the current point to avoid visiting again
        image[currentpos[1], currentpos[0]] = 0
        # find the matrix within which to search for the next point
        ystart = max([0, currentpos[1] - 1])
        ystop = min([height, currentpos[1] + 1])
        xstart = max([0, currentpos[0] - 1])
        xstop = min([width, currentpos[0] + 1])
        mtx = image[ystart : ystop + 1, xstart : xstop + 1]
        if np.sum(mtx) == 1:
            # if there is only one point to follow pick it as a next point 
            vals = np.argwhere(mtx == 1)
            currentpos = np.array([vals[0, 1] + xstart, vals[0, 0] + ystart])
            resultX.append(vals[0, 1] + xstart)
            resultY.append(vals[0, 0] + ystart)
        else:
            print('skeleton not prunned successfully!')
            break
        
        NoSteps += 1
    return np.array(resultX), np.array(resultY)

def interpolate_skeleton(x, y, npoints):
    s = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    s = s / s.max()
    S = np.zeros(len(x), dtype=np.float64)
    S[1:] = s
    Snew = np.linspace(0, 1, npoints)
    curve = np.concatenate([[x], [y]]).T
    myspline = interpolate.CubicSpline(S, curve)
    interp = myspline(Snew)
    return interp, S

def skeleton_endpoints(skel):
    # Make our input nice, possibly necessary.
    skel = skel.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    # Apply the convolution.
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)
    return np.array(np.where(filtered==11))

def correct_head_tail(skelX, skelY):
    N = skelX.shape[0]
    newX = np.nan * np.zeros_like(skelX)
    newY = np.nan * np.zeros_like(skelX)
    newX[0] = skelX[0]
    newY[0] = skelY[0]
    idx = np.where(~np.isnan(skelX[:, 0]))[0]
    #print(idx)
    prev = idx[0]
    newX[prev] = skelX[prev]
    newY[prev] = skelY[prev]

    for i in range(1, len(idx)):
        prev = idx[i - 1]
        ii = idx[i]
        distHH = np.sqrt((newX[prev, 0] - skelX[ii, 0])**2 + (newY[prev, 0] - skelY[ii, 0])**2)
        distTT = np.sqrt((newX[prev, -1] - skelX[ii, -1])**2 + (newY[prev, -1] - skelY[ii, -1])**2)
        distHT = np.sqrt((newX[prev, 0] - skelX[ii, -1])**2 + (newY[prev, 0] - skelY[ii, -1])**2)
        distTH = np.sqrt((newX[prev, -1] - skelX[ii, 0])**2 + (newY[prev, -1] - skelY[ii, 0])**2)
        if distHH + distTT > distHT + distTH:
            newX[ii] = skelX[ii, ::-1]
            newY[ii] = skelY[ii, ::-1]
        else:
            newX[ii] = skelX[ii]
            newY[ii] = skelY[ii]
    return newX, newY

def correct_head_tail_global_flip(Theta):
    # Head would be expected to have a higher curvature than the tail
    npts = Theta.shape[1]
    n3 = npts // 3
    flip = False
    HC = np.nanmean(abs(Theta[:, :n3]))
    TC = np.nanmean(abs(Theta[:, -n3:]))
    print('Head curvature ', HC)
    print('Tail curvature ', TC)
    
    if TC > HC:
        flip = True
        Theta = Theta[:, ::-1]
    return Theta, flip

def skel_length(skel):
    dx = np.diff(skel[:, 0])
    dy = np.diff(skel[:, 1])
    return np.sum(np.sqrt(dx**2 + dy**2))

def is_skel_good(skel, desired_length, deviation): 
    # check if the skeleton has desired length
    l = skel_length(skel)
    return l, np.abs(l - desired_length) < deviation

def skeletonise(mask, npoints, branch_thresh=40, skel_thresh=10):
    skeleton = cv2.ximgproc.thinning(mask)
    
    # find the longest path in the skeleton    
    fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=branch_thresh * u.pix, skel_thresh=skel_thresh * u.pix, prune_criteria='length')
    skeleton = fil.skeleton_longpath
    cv2.imshow('skeleton', skeleton)
    cv2.waitKey(100)
    endpoints = skeleton_endpoints(skeleton)
    startpos = np.array([endpoints[1, 0],endpoints[0, 0]])
    endpos = np.array([endpoints[1, 1], endpoints[0, 1]])
    skelX, skelY = find_skeleton(skeleton, startpos, endpos)
    interp, S = interpolate_skeleton(skelX, skelY, npoints)

    return interp, S

def skeletonise_array(ROI, npoints = 50, desired_length = 600, deviation=300, image=None):
    N, height, width = ROI.shape
    Frame = np.zeros((N, height, width), dtype=image.dtype)
    skelArray = np.nan * np.zeros(shape=(N, npoints, 2))
    is_skel_good = np.zeros(N, dtype=bool)
    l = 0
    for i in range(N):
        try:
            #retval, mask = cv2.threshold(ROI[i], 1, 255, cv2.THRESH_BINARY)
            mask = ROI[i].copy()
            cv2.imshow('mask', mask)
            interp, S = skeletonise(mask, npoints)
            if image is not None:
                frame = image[i].copy()
                pts = interp.reshape((-1, 1, 2))
                pts = pts.astype('int')
                cv2.polylines(frame, [pts], False, (0,0,0), 5)
                cv2.imshow('image', frame)
                Frame[i] = frame
            _, is_skel_good[i] = is_skel_good(interp, desired_length, deviation)
            skelArray[i] = interp
            print(str(i) + ' success')
        except KeyboardInterrupt:
                print('Interrupted')
                raise(KeyboardInterrupt)
        except Exception as error:
            print(error)
            print(str(i) + ' not skeletonised')
    skelX = skelArray[:, :, 0]
    skelY = skelArray[:, :, 1]
    skelX, skelY = correct_head_tail(skelX, skelY)
    skelArray[:, :, 0] = skelX
    skelArray[:, :, 1] = skelY
    cv2.destroyAllWindows()
    return skelArray, S, is_skel_good, Frame


def correct_unskeletonised(image, npoints, desired_length, deviation, denoising_strength, thresholdInit):
    N, h, w = image.shape
    print(N, ' skeletons to be corrected')
    Corrected = np.nan * np.zeros(shape=(N, npoints, 2))
    is_skel_good = np.zeros(N, dtype=bool)
    thList = range(thresholdInit - 3, thresholdInit + 10, 2)
    for j in range(N):
        print(j)
        frame = image[j]
        denoised = cv2.fastNlMeansDenoising(frame, None, denoising_strength, 7, 21)
        i = 0
        stop = False
        while (not stop) and (i < len(thList)):
            th = thList[i]
            ret, mask = cv2.threshold(denoised, th, 255, cv2.THRESH_BINARY)
            maskCol = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
            cv2.drawContours(maskCol, contours, 0, (205, 0, 140), thickness=4)
            cv2.imshow('mask Contours', maskCol)
            if len(contours) == 1:
                blur = cv2.GaussianBlur(mask, (0,0), sigmaX=9, sigmaY=9, borderType = cv2.BORDER_DEFAULT)
                result = skimage.exposure.rescale_intensity(blur, in_range=(90,255), out_range=(0,255))
                result = result.astype('uint8')
                retval, mask = cv2.threshold(result, 90, 255, cv2.THRESH_BINARY)
                try:
                    interp, S = skeletonise(mask, npoints)
                    _, is_skel_good[j] = is_skel_good(interp, desired_length, deviation)
                    # good we are done
                    if is_skel_good[j]:
                        print(j, 'corrected!')
                        Corrected[j] = interp
                        stop = True
                except KeyboardInterrupt:
                    print('Interrupted')
                    cv2.destroyAllWindows()
                    raise(KeyboardInterrupt)
                except Exception as error:
                    print(error)             
            elif len(contours) != 2:
                stop = True
            i += 1
    cv2.destroyAllWindows()
    return Corrected, is_skel_good


def load_skeletons(path, fname, npoints):
    # skeletons
    fnameCSV = os.path.join(path, 'skeletons', fname + '_skel.csv')
    SkelArray = np.loadtxt(fnameCSV).reshape(-1, npoints, 2)
    X0 = SkelArray[:, :, 0]
    Y0 = SkelArray[:, :, 1]
    
    N = SkelArray.shape[0]
    camSize = 6.5
    binning = 2
    magnification = 20
    pixToUm = camSize * binning / magnification
    
    # load stage data
    stageData = fname + '.txt'
    file = os.path.join(path, stageData)
    data = pd.read_csv(file, sep='\t', header=None)
    x = np.array(data[3])
    y = np.array(data[2])
    tS = np.array(data[1])

    # load the camera timestep 
    dataCam = pd.read_csv(os.path.join(path, fname + '_cam.txt'), sep='\t', header=None)
    tC = np.array(dataCam[1])

    x, y = match_stage_time_stamp(tS, x, y, tC, N)
    dx = np.append(np.diff(x),0)
    dy = np.append(np.diff(y),0)

    xpix = x / pixToUm
    ypix = y / pixToUm
    xpix = xpix - np.min(xpix)
    ypix = ypix - np.min(ypix)

    fps = np.mean(1 / np.diff(tC))

    xstage = xpix[:N]
    ystage = -ypix[:N]
    
    xnew = np.broadcast_to(np.expand_dims(xstage, axis=1), (N, npoints))
    ynew = np.broadcast_to(np.expand_dims(ystage, axis=1), (N, npoints))

    X = X0 + xnew
    Y = Y0 + ynew
    X, Y = correct_head_tail(X, Y)
    skeleton = np.zeros(shape = (N, npoints, 2))

    skeleton[:, :, 0] = X
    skeleton[:, :, 1] = Y
    
    return skeleton, fps

def calculate_curvature(X, Y, npoints, s=0.1):
    N, npts = X.shape
    npts2 = npoints
    Theta = np.nan * np.zeros((N, npts2))
    theta = np.zeros(npts - 1)
    theta2 = np.zeros(npts2)
    x = np.linspace(0, 1, npts-1)
    x2 = np.linspace(0, 1, npts2)
    for j in range(N):
        dx = np.diff(X[j])
        dy = np.diff(Y[j])    
        for i in range(npts-1):
            theta[i] = np.arctan2(dy[i], dx[i])
        theta = np.unwrap(theta)
        theta = theta - np.mean(theta)
        spl = splrep(x, theta, s=s)
        Theta[j] = splev(x2, spl)
    return Theta

def generateKdeDataFast(vx, vy, vz, xlims, ylims, zlims, clim, npts, fnameSave, cmap='GnBu', savepdf=False, in_notebook=True):
    # Ensure npts is a complex number to create a grid with npts steps
    npts_complex = complex(0, npts)
    
    # Generate grid
    X, Y, Z = np.mgrid[xlims[0]:xlims[1]:npts_complex, 
                       ylims[0]:ylims[1]:npts_complex, 
                       zlims[0]:zlims[1]:npts_complex]
    
    # Flatten the grid arrays and stack them
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    values = np.vstack([vx, vy, vz])

    # Compute density using Numba-optimized function
    density = compute_density(values, positions)
    density2 = np.reshape(density, X.shape)
    
    # Visualization with Pyvista
    grid = pv.StructuredGrid(X, Y, Z)
    grid["Density"] = density2.ravel()

    # Configure Pyvista to use an external window
    pv.set_plot_theme("document")  # Cleaner theme
    plotter = pv.Plotter(notebook=in_notebook)  # Set notebook=False for external window

    # Add volume rendering with a single color scale
    plotter.add_volume(grid, scalars="Density", clim=clim, cmap=cmap, opacity='sigmoid', opacity_unit_distance=1.0, show_scalar_bar=False)

    # Create x-y projection (z-axis shadow)
    xy_projection = np.sum(density2, axis=0)
    xy_grid = pv.StructuredGrid(X[:, :, 0], Y[:, :, 0], zlims[0] + np.zeros_like(X[:, :, 0]))
    xy_grid["Projection"] = xy_projection.ravel()
    plotter.add_mesh(xy_grid, scalars="Projection", cmap=cmap, clim=clim, opacity=0.75, show_scalar_bar=False)

    # Create x-z projection (y-axis shadow)
    xz_projection = np.sum(density2, axis=1)
    xz_grid = pv.StructuredGrid(X[:, 0, :], ylims[0] + np.zeros_like(Y[:, 0, :]), Z[:, 0, :])
    xz_grid["Projection"] = xz_projection.ravel()    
    plotter.add_mesh(xz_grid, scalars="Projection", cmap=cmap, clim=clim, opacity=0.75, show_scalar_bar=False)

    # Create y-z projection (x-axis shadow)
    yz_projection = np.sum(density2, axis=2)
    yz_grid = pv.StructuredGrid(xlims[0] + np.zeros_like(X[0, :, :]), Y[0, :, :], Z[0, :, :])
    yz_grid["Projection"] = yz_projection.ravel()
    
    plotter.add_mesh(yz_grid, scalars="Projection", cmap=cmap, clim=clim, opacity=0.75, show_scalar_bar=False)

    # Adjust plot settings to avoid unwanted artifacts
    val = 25
    plotter.camera_position = [(xlims[1] + val, ylims[1] + val, zlims[1] + val), (0, 0, 0), (0, 1, 0)]  # Move camera farther away
    #plotter.camera.zoom(3)  # Adjust zoom level (set to 1.2 to zoom out slightly)
    plotter.show_grid = False  # Disable grid lines
    plotter.show_axes = False  # Disable axes
    #plotter.show_axes = True 
    
    # Save and show plot
    plotter.show(screenshot=fnameSave + '.png')
    if savepdf:
        plotter.save_graphic(fnameSave + '.pdf')
    return density, density2, plotter


def generateKdeData2(vx, vy, vz, xlims, ylims, zlims, npts, fnameSave, cmap='GnBu'):
    import plotly.graph_objects as go
    import numpy as np
    from scipy import stats
    import os

    X, Y, Z = np.mgrid[xlims[0]:xlims[1]:npts, ylims[0]:ylims[1]:npts, zlims[0]:zlims[1]:npts]
    npts = int(np.imag(npts))  # Keep npts manageable
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    values = np.vstack([vx, vy, vz])

    # Adjust bandwidth here if needed
    kernel = stats.gaussian_kde(values, bw_method='scott')

    # Only calculate density on a reduced grid if npts is very large
    density = kernel(positions)
    density2 = np.reshape(density, X.shape)

    # Setup the volume rendering
    data = go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=density,
        isomin=0.001,
        isomax=np.max(density),
        opacity=0.4,
        surface_count=10,  # Reduce surface count for speed
        colorscale=cmap
    )

    r = np.linspace(xlims[0], xlims[1], npts)
    X, Y = np.meshgrid(r, r)

    # Create x-y projection (z-axis shadow)
    xy_projection = np.mean(density2, axis=2)
    xy_surface = go.Surface(z=zlims[0] + np.zeros_like(xy_projection), x=X, y=Y,
                            surfacecolor=xy_projection.T, opacity=0.4, showscale=False, colorscale=cmap)

    # Create x-z projection (y-axis shadow)
    xz_projection = np.mean(density2, axis=1)
    xz_surface = go.Surface(y=ylims[0] + np.zeros_like(xz_projection), x=X, z=Y,
                            surfacecolor=xz_projection.T, opacity=0.4, showscale=False, colorscale=cmap)

    # Create y-z projection (x-axis shadow)
    yz_projection = np.mean(density2, axis=0)
    yz_surface = go.Surface(x=xlims[0] + np.zeros_like(yz_projection), y=X, z=Y,
                            surfacecolor=yz_projection.T, opacity=0.4, showscale=False, colorscale=cmap)

    # Create the figure
    title = os.path.split(fnameSave)[1]
    fig = go.Figure(data=[data, xy_surface, xz_surface, yz_surface])
    fig.update_layout(scene=dict(aspectmode="cube",
                                 xaxis_title="e<sub>1</sub>",
                                 yaxis_title="e<sub>2</sub>",
                                 zaxis_title="e<sub>3</sub>",
                                 xaxis_showgrid=False,
                                 yaxis_showgrid=False,
                                 zaxis_showgrid=False),
                      title=title,
                      font=dict(
                          size=14,
                      )
                     )

    fig.write_image(fnameSave + '.png')
    fig.show()
    return data, density2


def plot_eigenworms(skeleton, plot, fnameSave, overwrite=True, npts=20j):
    path = r'W:\Ola\Upright tracker\Old\codes\model\behavioral-state-space\tosifahamed-behavioral-state-space-ad125e08db45\data'
    file = os.path.join(path, r'EigenWorms.mat')
    mtx = sio.loadmat(file)
    eig = mtx['EigenWorms']
    npts2 = eig.shape[0]
    X = skeleton[:, :, 0]
    Y = skeleton[:, :, 1]
    Theta = calculate_curvature(X, Y, 100, s=0.1)
    #Theta, flip = correct_head_tail_global_flip(Theta)
    #if flip:
    #    print('flipping head and tail')
    #    X = X[:, ::-1]
    #    Y = Y[:, ::-1]
    e1 = np.dot(eig[:, 0], Theta.T)
    e2 = np.dot(eig[:, 1], Theta.T)
    e3 = np.dot(eig[:, 2], Theta.T)

    Eig = np.concatenate((e1, e2, e3), axis=0)
    eigName = fnameSave + '_eig.npy'
    if os.path.isfile(eigName):
        if overwrite:
            os.remove(eigName)
    np.save(eigName, Eig)
    if plot:
        validIdx = ~np.isnan(e1)
        e1 = e1[validIdx]
        e2 = e2[validIdx]
        e3 = e3[validIdx]
        e1 = e1.reshape((1, -1))
        e2 = e2.reshape((1, -1))
        e3 = e3.reshape((1, -1))
        lims = [-12, 12]
        vmax2 = 0.0
        clim = None
        cmap = sns.cubehelix_palette(start=0.5, rot=-.75, as_cmap=True, light=0.97, gamma=1.2)
        density, density2, plotter = generateKdeDataFast(e1, e2, e3, lims, lims, lims, clim, npts, fnameSave, cmap=cmap, savepdf=True, in_notebook=True)

def normalise(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def circle3Points(x1, y1, x2, y2, x3, y3):
    Mx1, My1 = (x1 + x2) / 2, (y1 + y2) / 2
    Mx2, My2 = (x2 + x3) / 2, (y2 + y3) / 2
    m1 = -((x2 - x1) / (y2 - y1))
    m2 = -((x3 - x2) / (y3 - y2))
    h = (m1 * Mx1 - m2 * Mx2 + My2 - My1) / (m1 - m2)
    k = m1 * (h - Mx1) + My1
    r = np.sqrt((x1 - h)**2 + (y1 - k)**2)
    return h, k, r

def vector_decomposition2(v, e1, e2):
    coeff_e1 = np.dot(v, e1) / np.dot(e1, e1)
    coeff_e2 = np.dot(v, e2) / np.dot(e2, e2)
    result = np.array([coeff_e1, coeff_e2])
    return result

def findOrthogonalV(myVector):
    b = np.empty_like(myVector)
    b[0] = -myVector[1]
    b[1] = myVector[0]
    return b

def correct_dorso_ventral(x1, y1, x2, y2, x3, y3):
    cross_product = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
    return np.sign(cross_product)
    
def subtract_velocity(skeleton, pixToUm, fps):
    N, npts, _ = skeleton.shape
    Velocity = np.nan * np.zeros(shape=(N, npts, 2))
    VelocityUnits = np.nan * np.zeros(shape=(N, npts, 2, 2))
    for i in range(N-1):
        for j in range(0, npts):
            if (j == 0):
                jj = 1
            elif (j == (npts - 1)):
                jj = npts - 2
            else:
                jj = j
            x1, y1 = skeleton[i, jj - 1]
            x2, y2 = skeleton[i, jj]
            x3, y3 = skeleton[i, jj + 1]
            if ~np.isnan(x2):
                # find circum cirle passing by three points of the skeleton 
                h, k, r = circle3Points(x1, y1, x2, x2, x3, y3)
                if correct_dorso_ventral(x1, y1, x2, x2, x3, y3) < 0:
                    h = 2 * x2 - h
                    k = 2 * y2 - k
                # new vector x2 - h, y2 - k indicating the direction of normal velocity
                normal = normalise((x2 - h, y2 - k))
                trans =  normalise(findOrthogonalV(normal))
                dx2 = skeleton[i + 1, j, 0] - skeleton[i, j, 0]
                dy2 = skeleton[i + 1, j, 1] - skeleton[i, j, 1]
                vel = np.array([dx2, dy2]) * pixToUm * fps
                (vel_t, vel_n) = vector_decomposition2(vel, trans, normal)
                Velocity[i, j] = (vel_t, vel_n)
                VelocityUnits[i, j] = (trans, normal)
                
    return Velocity, VelocityUnits


def plot_midline(Skeleton, offset, save_name=None):

    fig, ax = plt.subplots()
    X = Skeleton[:, :, 0]
    Y = Skeleton[:, :, 1]
    N = X.shape[0]
    print(N)

    ln, = ax.plot(X[1], Y[1], 'k')
    ln0, = ax.plot(X[1, 0], Y[1, 0], 'ko')
    txt = ax.text(np.nanmin(X), np.nanmax(Y), str(offset[1, 0]) + ', ' + str(offset[1, 1]))
    ax.axis('off')
    ax.set_xlim(np.nanmin(X)-100, np.nanmax(X)+100)
    ax.set_ylim(np.nanmin(Y)-100, np.nanmax(Y)+100)
    ax.set_aspect('equal', 'box')
        
    def animate(i):
        ln.set_data(X[i], Y[i])
        ln0.set_data([X[i, 0]], [Y[i, 0]])
        txt.set_text(str(offset[i, 0]) + ', ' + str(offset[i, 1]))
        print(i)
        return ln, ln0, txt,
            
    anim = animation.FuncAnimation(fig, animate, frames=range(1, N-1), interval=500, blit=True)
    if save_name is not None:
        writer_video = animation.ImageMagickWriter(fps=10)
        anim.save(save_name, writer=writer_video)
        #plt.close()
    else:
        plt.show()
        plt.draw()
    print('done')

def visualise_velocity(skeleton, velocity, bodyPart, fname, fnameSave, height):
    fig, ax = plt.subplots()
    X = skeleton[:, :, 0]
    Y = skeleton[:, :, 1]
    N = X.shape[0]
    try:
        xls = fname + '.xlsx'
        df = pd.read_excel(xls, sheet_name=None)
        neurons = list(df.keys())
        colors = range(len(neurons))
        P = np.nan * np.zeros(shape = (N, len(neurons), 2))

        for inrn, neuron in enumerate(neurons):
            data = df[neuron]
            x = data['x']
            y = data['y']
            y = height - y
            P[:, inrn, 0] = x
            P[:, inrn, 1] = y
    except:
        neurons = ['a']
        colors = range(len(neurons))
        P = np.zeros((N, 1, 2))
        

    ln, = ax.plot(X[0], Y[0], 'k')
    vn = ax.quiver(X[0, 1], Y[0, 1], 0, 1, color='r', angles='xy', scale_units='xy', scale=1.)
    vt = ax.quiver(X[0, 1], Y[0, 1], 1, 0, color='b', angles='xy', scale_units='xy', scale=1.)
    v = ax.quiver(X[0, 1], Y[0, 1], 1, 1, color='g', angles='xy', scale_units='xy', scale=1.)
    pt = ax.scatter(np.ones(len(neurons)), np.ones(len(neurons)), c=colors)
        
    try:
        handles, labels = pt.legend_elements(prop="colors")
        legend = ax.legend(handles, neurons, loc="upper right", title="Neurons")
        #ax.legend()
    except:
        pass

    ax.set_xlim(np.nanmin(X), np.nanmax(X))
    ax.set_ylim(np.nanmin(Y), np.nanmax(X))
    ax.set_aspect('equal', 'box')
    factor = 2
    j = bodyPart
    
    def animate(i):
        ln.set_data(X[i], Y[i])
        #three body points: x1, y1, x2, x2, x3, y3
        x1, y1 = skeleton[i, j - 1]
        x2, y2 = skeleton[i, j]
        x3, y3 = skeleton[i, j + 1]
        
        h, k, r = circle3Points(x1, y1, x2, x2, x3, y3)
        # new vector x2 - h, y2 - k indicating the direction of normal velocity
        normal = normalise((x2 - h, y2 - k)) 
        trans = normalise(findOrthogonalV(normal))
        (vel_t, vel_n) = velocity[i, j]
        n = vel_n * normal
        t = vel_t * trans
        try:
            vn.set_offsets(np.array((x2, y2)))
            vn.set_UVC(factor * n[0] , factor * n[1])
            
            vt.set_offsets(np.array((x2, y2)))
            vt.set_UVC(factor * t[0], factor * t[1])
            
            v.set_offsets(np.array((x2, y2)))
            v.set_UVC(factor * (n[0] + t[0]), factor * (n[1] + t[1]))

        except:
            pass
       
        offset = [(x, y) for (x, y) in P[i]]
        pt.set_offsets(offset)
        return ln, vn, vt, pt
            
    anim = animation.FuncAnimation(fig, animate, frames=N-1, interval=400, blit=True)
    
    # saving animation to gif
    if fnameSave != '':
        writer_video = animation.ImageMagickWriter(fps=10)
        fnameSave2 = fnameSave + '_vectors.gif'
        anim.save(fnameSave2, writer=writer_video)
    plt.close()
    print('done')
    
def correct_head_tail_neural_data(skeleton, fname, height):
    skeletonNew = np.nan * np.zeros_like(skeleton)
    N, npts, dim = skeleton.shape

    xls = fname + '.xlsx'
    df = pd.read_excel(xls, sheet_name=None)
    neurons = list(df.keys())
    colors = np.random.rand(len(neurons))
    P = np.nan * np.zeros(shape = (N, len(neurons), 2))
    BodyIndex = np.nan * np.zeros(len(neurons))
    for inrn, neuron in enumerate(neurons):
        data = df[neuron]
        if 'x' in data:
            x = data['x']
            y = data['y']
            y = height - y
            P[:, inrn, 0] = x
            P[:, inrn, 1] = y
        if (neuron == 'ALM') or (neuron == 'AVM'):
            BodyIndex[inrn] = 0.4 * npts
        elif (neuron == 'PVM'):
            BodyIndex[inrn] = 0.7 * npts
        elif (neuron == 'PLM'):
            BodyIndex[inrn] = 1 * npts
    nanIdxs = np.isnan(np.nansum(P[:, :, 0], axis=1) + np.sum(skeleton[:, :, 0], axis=1))  
    validIdxs = np.where(~nanIdxs)[0]
    for i in range(len(validIdxs)):
        idx = validIdxs[i]
        # find the initial body index for each neuron (based on the first frame? or neuron names)
        # for i-th frame check if the closest body index is closer to current or flipped skeleton
        cost = [0, 0]
        for inrn, neuron in enumerate(neurons):
            #current distance to each body point
            d = np.sqrt((skeleton[idx, :, 0] - P[idx, inrn, 0])**2 + (skeleton[idx, :, 1] - P[idx, inrn, 1])**2)
            # the closest body point
            argmin = np.argmin(d)        
            cost[0] = np.nansum([cost[0], (BodyIndex[inrn] - argmin)**2])
            cost[1] = np.nansum([cost[1], (BodyIndex[inrn] - (npts - argmin))**2])
        if cost[1] < cost[0]:
            skeletonNew[idx] = skeleton[idx, ::-1]
        else:
            skeletonNew[idx] = skeleton[idx]
    lastValid = validIdxs[0]
    for i in range(N):
        if i not in validIdxs:
            distHH = np.sqrt((skeletonNew[lastValid, 0, 0] - skeleton[i, 0, 0])**2 + (skeletonNew[lastValid, 0, 1] - skeleton[i, 0, 1])**2)
            distTT = np.sqrt((skeletonNew[lastValid, -1, 0] - skeleton[i, -1, 0])**2 + (skeletonNew[lastValid, -1, 1] - skeleton[i, -1, 1])**2)
            distHT = np.sqrt((skeletonNew[lastValid, 0, 0] - skeleton[i, -1, 0])**2 + (skeletonNew[lastValid, 0, 1] - skeleton[i, -1, 1])**2)
            distTH = np.sqrt((skeletonNew[lastValid, -1, 0] - skeleton[i, 0, 0])**2 + (skeletonNew[lastValid, -1, 1] - skeleton[i, 0, 1])**2)
            if distHH + distTT > distHT + distTH:
                skeletonNew[i] = skeleton[i, ::-1]
            else:
                skeletonNew[i] = skeleton[i]
        if not np.isnan(np.sum(skeleton[i])):
            lastValid = i
    return skeletonNew