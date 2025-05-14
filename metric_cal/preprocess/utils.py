import os
import torch
import h5py
import matplotlib.pyplot as plt
import ants
import numpy as np
import pickle

def normalize_percentile(img, lower_percentile=1, upper_percentile=99.9, clip=True):
    """ Normalization to the lower and upper percentiles 
        Utility functions from:
        https://github.com/melanieganz/ImageQualityMetricsMRI/blob/main/utils/data_utils.py

    """
    img = img.astype(np.float32)
    lower = np.percentile(img, lower_percentile)
    upper = np.percentile(img, upper_percentile)
    img = (img - lower) / (upper - lower)
    if clip:
        img = np.clip(img, 0, 1)
    return img

def sort_out_zero_slices(img, ref, brainmask=None,non_zero_ratio=0.01):
    """ Only keep slices with more than 10% non-zero values in img and ref. 
        Utility functions from:
        https://github.com/melanieganz/ImageQualityMetricsMRI/blob/main/utils/data_utils.py

    """
    # non_zero_ratio = 0.01
    zero_slices_img = np.where(np.sum(img > 0, axis=(0, 2)) / img[1].size < non_zero_ratio)[0]
    # print(zero_slices_img)
    # print(img.shape,ref.shape)
    if ref is not None:
        zero_slices_ref = np.where(np.sum(ref > 0, axis=(0, 2)) / ref[1].size < non_zero_ratio)[0]
        zero_slices = np.unique(np.concatenate((zero_slices_img, zero_slices_ref)))
        ref = np.delete(ref, zero_slices, axis=1)
    else:
        zero_slices = zero_slices_img
    # print(zero_slices)
    img = np.delete(img, zero_slices, axis=1)
    if brainmask is not None:
        brainmask = np.delete(brainmask, zero_slices, axis=1)
        return img, ref, brainmask
    else:
        return img, ref
