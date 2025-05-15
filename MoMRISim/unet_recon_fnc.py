from MoMRISim.util.helpers_math import complex_abs,normalize_separate_over_ch
from MoMRISim.util.unet import Unet
import torch
import ants

# Using the NN to reconstruct the images:
def Unet_recon(
        input_img,
        binary_background_mask,
        args,
):
    # Load the U-Net weight from the huggingface:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    weights_path = hf_hub_download(
        repo_id="mli-lab/Unet48-2D-CC359",
        filename="model.safetensors"
    )
    state_dict = load_file(weights_path)
    model = Unet(in_chans=2, out_chans=2, chans=48,)
    model.load_state_dict(state_dict)
    model.to(f"cuda:{args.gpu}")
    model.eval()
    print(input_img.shape)
    input_img_2D, mean, std = normalize_separate_over_ch(input_img.moveaxis(-1,1), eps=1e-11)
    recon_image_full_1c = torch.zeros_like(input_img_2D)
    with torch.no_grad():
        for i in range(len(input_img_2D)):
            recon_image_full_1c[i] = model(input_img_2D[i].unsqueeze(0)).cpu()
    recon_image_full_1c = recon_image_full_1c * std + mean
    recon_image_full_1c = torch.moveaxis(recon_image_full_1c, 1, -1)
    recon_image_full_1c = recon_image_full_1c.cuda(args.gpu) * binary_background_mask
    return complex_abs(recon_image_full_1c)

def registration(recon, target):
    if recon.shape[-1] == 2:
        recon = complex_abs(recon).squeeze().cpu().numpy()
    else:
        recon = recon.squeeze().cpu().numpy()
    if target.shape[-1] == 2:
        target = complex_abs(target).squeeze().cpu().numpy()
    else:
        target = target.squeeze().cpu().numpy()
    recon_image_fg_1c_ants = ants.from_numpy(recon)
    target_image_fg_1c_ants = ants.from_numpy(target)
    print(recon.shape, target.shape)
    mytx = ants.registration(fixed=target_image_fg_1c_ants , moving=recon_image_fg_1c_ants, 
                                    grad_step=0.1,
                                    aff_random_sampling_rate=0.2,
                                    singleprecision=False,
                                    aff_iterations=(2100, 1200, 1200, 100),
                                    type_of_transform="DenseRigid",
                                    )

    recon_image_fg_1c_aligned = ants.apply_transforms(fixed=target_image_fg_1c_ants, moving=recon_image_fg_1c_ants,
                                        transformlist=mytx['fwdtransforms'],
                                        interpolator='lanczosWindowedSinc').numpy()
    return recon_image_fg_1c_aligned