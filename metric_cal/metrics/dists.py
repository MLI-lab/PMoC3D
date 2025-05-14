import torch
from DISTS_pytorch import DISTS

def dists_score(recon: torch.Tensor, 
         ref: torch.Tensor, 
         score_direction: int, 
         gpu: int=-1) -> float:
    """
    Calculate the dists score between two images under a certain direction.

    Args:
        recon (torch.Tensor): Reconstruction tensor.
        ref (torch.Tensor): Reference tensor.
    Returns:
        float: score value
    """
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")
    recon = recon.moveaxis(score_direction,0).unsqueeze(1).repeat(1,3,1,1)
    ref = ref.moveaxis(score_direction,0).unsqueeze(1).repeat(1,3,1,1)
    dists_fnc = DISTS().to(device)
    dists_score = dists_fnc(recon.to(device),ref.to(device)).detach().cpu()
    return dists_score.mean().item()