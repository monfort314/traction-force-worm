# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4, 2023

@author: apidde
"""

import os
import skimage.exposure
import cv2
import numpy as np

def smooth(x, N, mode='same', fixEnds = True):
    A = np.convolve(x, np.ones(N)/N, mode='same')
    if fixEnds:
        for i in range(int((N + 1)/2)):
            A[i] = np.mean(x[: 2 * i + 1])
            A[-i - 1] = np.mean(x[-2 * i -1:])
    return A

def mask_worm(image, min_worm_size, kernel, show):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask0 = np.zeros_like(image)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_worm_size:
            cv2.drawContours(mask0, [cnt], 0, (255, 255, 255), thickness=cv2.FILLED)                
    dilation = cv2.dilate(mask0, kernel, iterations = 10)
    blur = cv2.GaussianBlur(mask0, (0,0), sigmaX=9, sigmaY=9, borderType = cv2.BORDER_DEFAULT)
    result = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))
    result = result.astype('uint8')
    retval, mask = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
    if show:
        cv2.imshow('mask', mask)
        cv2.waitKey(1)
    return mask

def set_threshold(frame, thresh_list, denoising_strength=20, manual=True, wormpercent=0.05):
    percent = np.zeros(len(thresh_list))
    for i in range(len(thresh_list)):
        th = thresh_list[i]
        denoised = cv2.fastNlMeansDenoising(frame, None, denoising_strength, 7, 21)
        ret, mask = cv2.threshold(denoised, th, 255, cv2.THRESH_BINARY)
        percent[i] = np.mean(mask) / 255
        print(th, ', No pix ', np.sum(mask) / 255, ', fraction of the view: ', percent[i])
        cv2.imshow('threshold ' + str(th), mask)
    if manual:
        k = cv2.waitKey(0)
        newTh = int(input('Select the threshold: '))
    else:
        cv2.waitKey(10)
        newTh = thresh_list[np.where(percent >= wormpercent)[0][-1]]
        print('automatically selected threshold ', newTh)
    cv2.destroyAllWindows()
    return newTh

def identify_worm(image, from_threshold, threshold=15, min_worm_size=10000, denoising_strength=20, edge_threshold=(4, 15), morph_kernel_size=8, iterations_dilation=6, show=True):
    N, height, width = image.shape
    ROI = np.zeros(image.shape, dtype='uint8')
    #BCKG = np.zeros_like(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morph_kernel_size, morph_kernel_size))
    th = threshold
    for i in range(N):
        frame = image[i]
        denoised = cv2.fastNlMeansDenoising(frame, None, denoising_strength, 7, 21) 
        if from_threshold:
            if type(threshold) == np.ndarray:
                th = threshold[i]
            ret, mask = cv2.threshold(denoised, th, 255, cv2.THRESH_BINARY)
            dilation = cv2.dilate(mask, kernel, iterations = iterations_dilation)
            #cv2.imshow('from Threshold', mask)
            #cv2.imshow('dilation', dilation)
        else:
            edges = cv2.Canny(image = denoised, threshold1 = edge_threshold[0], threshold2 = edge_threshold[1]) 
            dilation = cv2.dilate(edges, kernel, iterations = iterations_dilation)
            #cv2.imshow('edges', edges)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations = 3)
        mask0 = maskWorm(closing, min_worm_size, kernel, show)
        roi = cv2.bitwise_and(mask0, frame) 
        ROI[i] = mask0
        if show:
            cv2.imshow('mask', mask0)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    return ROI