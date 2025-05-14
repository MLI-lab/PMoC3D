import torch
import lpips

def lpips_score(recon: torch.Tensor, 
         ref: torch.Tensor, 
         score_direction: int, 
         gpu: int=-1) -> float:
    """
    Calculate the lpips score between two images under a certain direction.

    Args:
        recon (torch.Tensor): Reconstruction tensor.
        ref (torch.Tensor): Reference tensor.
    Returns:
        float: score value
    """
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")
    recon = recon*2-1
    ref = recon*2-1
    recon = recon.moveaxis(score_direction,0).unsqueeze(1).repeat(1,3,1,1)
    ref = ref.moveaxis(score_direction,0).unsqueeze(1).repeat(1,3,1,1)
    lpips_fnc = lpips.LPIPS(net='vgg').to(device)
    lpips_score = lpips_fnc(recon.to(device),ref.to(device)).detach().cpu()
    return lpips_score.mean().item()