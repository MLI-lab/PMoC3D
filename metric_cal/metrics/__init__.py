from .psnr import psnr
from .ssim import ssim
from .AP import AP
from .lpips import lpips_score
from .dists import dists_score as dists
from .dreamsim import dreamsim_score as dreamsim
from .aes import aes_score as aes
from .tg import tg_score as tg
from .momrisim import momrisim

__all__ = [
    "psnr",
    "ssim",
    "AP",
    "lpips",
    "dists",
    "dreamsim",
    "aes",
    "tg",
    "momrisim"
]


metrics = {
    'psnr': psnr,
    'ssim': ssim,
    'ap': AP,
    'lpips': lpips_score,
    'dists': dists,
    'dreamsim': dreamsim,
    'aes': aes,
    'tg': tg,
    'momrisim': momrisim
}