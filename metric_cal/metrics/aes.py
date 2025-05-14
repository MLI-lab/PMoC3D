import torch
import numpy as np
from scipy.ndimage import convolve
from skimage.feature import canny

def aes_score(recon: torch.Tensor, score_direction: int, ) -> float:
    """
    Calculate the Average Edge Strength(AES) of the reconstruction.

    Args:
        recon (torch.Tensor): Reconstruction tensor.
    Returns:
        float: AES value in decibels (dB).
    """
    recon = recon.squeeze().moveaxis(score_direction,0)
    return aes(recon.numpy())

def aes(img, brainmask=None):
    """
    Calculate the metric Average Edge Strength.
    This implementation is from: 
    https://github.com/melanieganz/ImageQualityMetricsMRI/blob/main/metrics/gradient_metrics.py
    which in turn is based on the original implementation by Simon Chemnitz-Thomsen.

    Reference:
    Quantitative framework for prospective motion correction evaluation
    Nicolas Pannetier, Theano Stavrinos, Peter Ng, Michael Herbst,
    Maxim Zaitsev, Karl Young, Gerald Matson, and Norbert Schuff

    Parameters
    ----------
    img : numpy array
        Image for which the metrics should be calculated.
    brainmask : numpy array
        Brainmask for the image. If provided, the metric will be calculated
        only on the masked region.
    sigma : float
        Standard deviation of the Gaussian filter used
        during canny edge detection.
    n_levels : int
        Levels of intensities to bin image by
    bin : bool
        Whether to bin the image
    crop : bool
        Whether to crop image/ delete empty slices
    weigt_avg : bool
        Whether to calculate the weighted average (depending on the
        proportion of non-zero pixels in the slice).
    reduction : str
        Method to reduce the edge strength values.
        'mean' or 'worst'

    Returns
    -------
    AES : float
        Average Edge Strength measure of the input image.
    """
    sigma=np.sqrt(2)
    # Centered Gradient kernel in the y- and x-direction
    y_kern = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]])
    x_kern = y_kern.T
    # Empty array to contain edge strenghts
    # Function returns the mean of this list
    es = []
    img = img.astype(float)

    for sl in range(img.shape[0]):
        # Slice to do operations on
        im_slice = img[sl]
        # Convolve slice
        x_conv = convolve(im_slice, x_kern)
        y_conv = convolve(im_slice, y_kern)
        # Canny edge detector
        canny_img = canny(im_slice, sigma=sigma)
        if brainmask is not None:
            canny_img = np.ma.masked_array(canny_img,
                                           mask=(brainmask[sl] != 1))
            x_conv = np.ma.masked_array(x_conv,
                                        mask=(brainmask[sl] != 1))
            y_conv = np.ma.masked_array(y_conv,
                                        mask=(brainmask[sl] != 1))
        # Numerator and denominator, to be divided
        # defining the edge strength of the slice
        numerator = np.sum(canny_img * (x_conv ** 2 + y_conv ** 2))
        denominator = np.sum(canny_img)
        # Append the edge strength
        es.append(np.sqrt(numerator) / denominator)

    es = np.array(es)
    # Remove nans
    es = es[~np.isnan(es)]
    # return np.mean(es)
    return np.min(es)