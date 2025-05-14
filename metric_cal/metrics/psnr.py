import torch

def psnr(recon: torch.Tensor, ref: torch.Tensor) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        recon (torch.Tensor): First image tensor.
        ref (torch.Tensor): Second image tensor.
    Returns:
        float: PSNR value in decibels (dB).
    """
    mse = torch.mean(((recon - ref)**2))
    psnr = 20 * torch.log10(torch.tensor(ref.max().item()))- 10 * torch.log10(mse)
    return psnr.item()