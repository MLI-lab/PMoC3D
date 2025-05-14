import os
import torch
import h5py
import matplotlib.pyplot as plt
import ants
import numpy as np
import pickle

def align_volumes_with_ants(
        recon_image_fg_1c: np.ndarray, # of shape (x,y,z)
        target_image_fg_1c: np.ndarray, # of shape (x,y,z)
):
    # recon_image_fg_1c = normalize_percentile(recon_image_fg_1c)
    # target_image_fg_1c = normalize_percentile(target_image_fg_1c)
    
    recon_image_fg_1c_ants = ants.from_numpy(recon_image_fg_1c)
    target_image_fg_1c_ants = ants.from_numpy(target_image_fg_1c)

    mytx = ants.registration(fixed=target_image_fg_1c_ants , moving=recon_image_fg_1c_ants, 
                            grad_step=0.1,
                            aff_random_sampling_rate=0.2,
                            singleprecision=False,
                            aff_iterations=(2100, 1200, 1200, 100),
                            type_of_transform="DenseRigid",
                            mask_all_stages = True,
                            )

    recon_image_fg_1c_aligned = ants.apply_transforms(fixed=target_image_fg_1c_ants, moving=recon_image_fg_1c_ants,
                                      transformlist=mytx['fwdtransforms'],
                                      interpolator='lanczosWindowedSinc').numpy()

    return recon_image_fg_1c_aligned