import torch
from dreamsim import dreamsim
from PIL import Image
import numpy as np

def dreamsim_score(recon: torch.Tensor, 
         ref: torch.Tensor, 
         score_direction: int, 
         gpu: int=-1) -> float:
    """
    Calculate the DreamSim score between two images under a certain direction.

    Args:
        recon (torch.Tensor): Reconstruction tensor.
        ref (torch.Tensor): Reference tensor.
    Returns:
        float: score value
    """
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")
    model, preprocess = dreamsim(pretrained=True, device=device)
    num_slices = recon.shape[score_direction]

    processed_recon = []
    processed_ref = []
    for i in range(num_slices):
        if score_direction == 0:
            recon_slice = recon[i, :, :]
            ref_slice   = ref[i, :, :]
        elif score_direction == 1:
            recon_slice = recon[:, i, :]
            ref_slice   = ref[:, i, :]
        else:
            recon_slice = recon[:, :, i]
            ref_slice   = ref[:, :, i]
        recon_tensor = torch.from_numpy(recon_slice).unsqueeze(0).repeat(3, 1, 1)
        ref_tensor   = torch.from_numpy(ref_slice).unsqueeze(0).repeat(3, 1, 1)
        recon_img = Image.fromarray((recon_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        ref_img   = Image.fromarray((ref_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        proc_recon = preprocess(recon_img)
        proc_ref   = preprocess(ref_img)
        processed_recon.append(proc_recon.squeeze())
        processed_ref.append(proc_ref.squeeze())

    recon = torch.stack(processed_recon, dim=0).cuda(gpu)
    ref   = torch.stack(processed_ref, dim=0).cuda(gpu)
    dreamsim_score = model(recon, ref)
    return dreamsim_score.mean().item()