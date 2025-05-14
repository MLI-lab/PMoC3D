from .psnr import psnr
from .ssim import ssim
from .AP import AP
from .lpips import lpips_score as lpips
from .dists import dists_score as dists
from .dreamsim import dreamsim_score as dreamsim
from .aes import aes_score as aes
from .tg import tg_score as tg

__all__ = [
    "psnr",
    "ssim",
    "AP",
    "lpips",
    "dists",
    "dreamsim",
    "aes",
    "tg"
]


metrics = {
    'psnr': psnr,
    'ssim': ssim,
    'ap': AP,
    'lpips': lpips,
    'dists': dists,
    'dreamsim': dreamsim,
    'aes': aes,
    'tg': tg
}