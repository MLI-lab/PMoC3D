import ants
import numpy as np
from .utils import normalize_percentile

def align_volumes_with_ants(
        recon_image_fg_1c: np.ndarray, # of shape (x,y,z)
        target_image_fg_1c: np.ndarray, # of shape (x,y,z)
        use_nonrigdid: bool = False,
):
    recon_image_fg_1c = normalize_percentile(recon_image_fg_1c)
    target_image_fg_1c = normalize_percentile(target_image_fg_1c)
    
    recon_image_fg_1c_ants = ants.from_numpy(recon_image_fg_1c)
    target_image_fg_1c_ants = ants.from_numpy(target_image_fg_1c)
    
    if use_nonrigdid:
        mytx = ants.registration(fixed=target_image_fg_1c_ants , moving=recon_image_fg_1c_ants, 
                            grad_step=0.1,
                            aff_random_sampling_rate=0.2,
                            singleprecision=False,
                            aff_iterations=(2100, 1200, 1200, 100),
                            type_of_transform="SyN",
                            )
    else:
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