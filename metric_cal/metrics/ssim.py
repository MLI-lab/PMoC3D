import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Union

def ssim(recon: torch.Tensor, 
         ref: torch.Tensor, 
         score_direction: int, 
         gpu: int=-1,
         data_range: Optional[Union[float, torch.Tensor]] = None ) -> float:
    """
    Calculate the artifact power (AP) between two images.

    Args:
        recon (torch.Tensor): Reconstruction tensor.
        ref (torch.Tensor): Reference tensor.
    Returns:
        float: AP value.
    """
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")
    # # SSIM
    ssim_loss = SSIMLoss(gpu=gpu)
    recon = recon.unsqueeze(0).moveaxis(score_direction+1,0)
    ref = ref.unsqueeze(0).moveaxis(score_direction+1,0)
    if data_range is None:
        data_range = torch.max(ref).to(device).repeat(ref.shape[0])
    ssim = 1-ssim_loss(recon.to(device), ref.to(device), data_range=data_range).item()
    return ssim

class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, gpu: int = -1):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")
        self.win_size = win_size
        self.k1, self.k2 = torch.tensor(k1).to(device), torch.tensor(k2).to(device)
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size).to(device) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = torch.tensor(NP / (NP - 1)).to(device)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()