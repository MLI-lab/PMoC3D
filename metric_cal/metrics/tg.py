import torch
import numpy as np
from scipy.ndimage import sobel

def tg_score(recon: torch.Tensor, score_direction: int, ) -> float:
    """
    Calculate the Average Edge Strength(AES) of the reconstruction.

    Args:
        recon (torch.Tensor): Reconstruction tensor.
    Returns:
        float: AES value in decibels (dB).
    """
    recon = recon.squeeze().moveaxis(score_direction,0)
    return tenengrad(recon.numpy())


def calc_gradient_magnitude(img, mode="2d"):
    """Calculate the magnitude of the image gradient.
    This implementation is from: 
    https://github.com/melanieganz/ImageQualityMetricsMRI/blob/main/metrics/gradient_metrics.py

        Note:
        - The image is assumed to be a 3D image.
        - The image is assumed to be masked and normalised to [0, 1].
        - The image is converted to floating point numbers for a correct
        calculation of the gradient.
    """

    img = img.astype(float)

    grad_x = sobel(img, axis=1, mode='reflect')
    grad_y = sobel(img, axis=2, mode='reflect')

    if mode == "2d":
        return np.sqrt(grad_x ** 2 + grad_y ** 2)
    elif mode == "3d":
        grad_z = sobel(img, axis=0, mode='reflect')
        return np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
    else:
        raise ValueError(f"Mode {mode} not supported.")


def tenengrad(img, brainmask=None):
    """Tenengrad measure of the input image.
    This implementation is from: 
    https://github.com/melanieganz/ImageQualityMetricsMRI/blob/main/metrics/gradient_metrics.py
    which in turn is based on the article:
    Krotkov E. Focusing. Int J Comput Vis. 1988; 1(3):223-237

    Parameters
    ----------
    img : numpy array
        image for which the metrics should be calculated.
    brainmask : boolean True or False, optional
        If True, a brainmask was used to mask the images before 
        calculating the metrics. Image is flattened prior metric 
        estimation. The default is False.

    Returns
    -------
    tg : float
        Tenengrad measure of the input image.
    """

    grad = calc_gradient_magnitude(img, mode="2d")

    if brainmask is not None:
        grad = np.ma.masked_array(grad, mask=(brainmask != 1))

    # return np.mean(grad ** 2)
    grad_slices = np.mean(grad ** 2, axis=(1, 2))
    return np.min(grad_slices)