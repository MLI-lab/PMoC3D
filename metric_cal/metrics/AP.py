import torch

def AP(recon: torch.Tensor, ref: torch.Tensor) -> float:
    """
    Calculate the artifact power (AP) between two images.

    Args:
        recon (torch.Tensor): Reconstruction tensor.
        ref (torch.Tensor): Reference tensor.
    Returns:
        float: AP value.
    """
    ap_score = torch.sum(torch.abs(torch.abs(recon) - torch.abs(ref))**2) / torch.sum(torch.abs(ref)**2)
    return ap_score.item()