# %% Load the training file:
import pickle
import os
import h5py
import torch
from matplotlib import pyplot as plt
import random
import numpy as np
from types import SimpleNamespace
from MoMRISim.util.helpers_math import complex_abs, complex_mul, complex_conj, ifft2c_ndim, fft2c_ndim


from MoMRISim.motion_simulation.motion_forward_backward_models import motion_correction_NUFFT, motion_corruption_NUFFT
from MoMRISim.motion_simulation.sampling_trajectories import sim_motion_get_traj
from MoMRISim.motion_simulation.motion_trajectories import gen_rand_mot_params_interShot

from MoMRISim.L1min_Simple import L1minModuleBase
from MoMRISim.unet_recon_fnc import Unet_recon, registration

# Parameter design:
motion_severity_settings = {
    1: {"num_event":1, "max_mot":2},
    2: {"num_event":5, "max_mot":2},
    3: {"num_event":10, "max_mot":2},
    4: {"num_event":1, "max_mot":5},
    5: {"num_event":1, "max_mot":10},
    6: {"num_event":5, "max_mot":5},
    7: {"num_event":10, "max_mot":5},
    8: {"num_event":5, "max_mot":10},
    9: {"num_event":10, "max_mot":10},
}
args = SimpleNamespace(
    gpu = 0,
)
args.dataset_save_path = "/media/ssd0/kun/MoMRISim_dataset"
volume_info_path = "./MoMRISim/dataset/volume_dataset_freqEnc170_train_len40.pickle"
mask3D_path = "./MoMRISim/mask_3D_size_218_170_256_R4.9_poisson_simreal.pickle"
train_base_path = "/media/ssd0/cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Train_converted/"
train_smaps_base_path = "/media/ssd0/cc-359_raw/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Train_s_maps_3D/"
args.nufft_norm = "ortho"
args.Ns = 52
args.sampTraj_simMot = "random_cartesian"
args.center_in_first_state = True
args.random_sampTraj_seed = 10086

# Load the volume information:
with open(volume_info_path, 'rb') as f:
    volume_info = pickle.load(f)
# Print some information about the dataset"
len_dataset = len(volume_info)
print("Number of volumes in the training set: ", len_dataset)
print("Example name of the first volume: ", volume_info[0]['filename'])
# Mask the kspace generation:
with open(mask3D_path, 'rb') as f:
    mask3D = pickle.load(f)
mask3D = torch.tensor(mask3D).unsqueeze(0).unsqueeze(-1).cuda(args.gpu)
# Load the reference volume: 
for picked_volume_index in range(len(volume_info)):
    picked_volume_info = volume_info[picked_volume_index]['filename']
    # Load the volume:
    with h5py.File(os.path.join(train_base_path, picked_volume_info), 'r') as f:
        # Print the keys of the file:
        kspace3D = f['kspace'][:]
        print("Kspace shape: ", kspace3D.shape)
    with h5py.File(os.path.join(train_smaps_base_path, "smaps_"+picked_volume_info), 'r') as f:
        # Print the keys of the file:
        smaps3D = f['smaps'][:]
        print("smaps shape: ", kspace3D.shape)
    # Generate masked images:
    kspace3D = torch.tensor(kspace3D).cuda(args.gpu)
    smaps3D = torch.tensor(smaps3D).cuda(args.gpu)
    smaps3D_conj = complex_conj(smaps3D)
    kspace3D_masked = kspace3D * mask3D
    binary_background_mask = torch.round(torch.sum(complex_mul(complex_conj(smaps3D).cuda(args.gpu),smaps3D.cuda(args.gpu)),0)[:,:,:,0:1])
    # Using L1 reconstruction to reconstruct the reference volume:
    traj = sim_motion_get_traj(args, mask3D)
    L1_args = SimpleNamespace(
            gpu = args.gpu,
            nufft_norm = args.nufft_norm,
            )
    recon_model = L1minModuleBase(
            smaps3D=smaps3D,
            binary_background_mask=binary_background_mask,
            masked_corrupted_kspace3D=kspace3D_masked,
            mask3D=torch.tensor(mask3D).cuda(args.gpu),
            gpu=args.gpu,                             
            L1min_lr=5e7,          
            L1min_lambda=1e-3,
            L1min_num_steps=50,
            traj=traj,
            pred_motion_params=None,  
            L1min_optimizer_type="SGD",
            args=L1_args
        )
    reference_image = recon_model.run_L1min()
    # Create the folder for saving the volumes:
    path = os.path.join(args.dataset_save_path, picked_volume_info)
    os.makedirs(path, exist_ok=True)
    # Save the volumes: 
    with h5py.File(os.path.join(path, "ref_img3D.h5"), 'w') as f:
        # Make the key as reference:
        f.create_dataset("reference", data=complex_abs(reference_image).squeeze().cpu().numpy())
    # Load the reference image:
    path = os.path.join(args.dataset_save_path, picked_volume_info)
    with h5py.File(os.path.join(path, "ref_img3D.h5"), 'r') as f:
        # Print the keys of the file:
        reference_image = f['reference'][:]
        print("Reference image shape: ", reference_image.shape)
    reference_image = torch.tensor(reference_image)
    # Choosing 2 motion severity:
    choosing_severity_level = np.arange(1, 10)
    np.random.shuffle(choosing_severity_level)
    choosing_severity_level = choosing_severity_level[:7]
    for severity in choosing_severity_level:
        args.Ns = 52
        args.motionTraj_simMot = "uniform_interShot_event_model"
        args.sampTraj_simMot = "random_cartesian"
        args.center_in_first_state = True
        args.random_sampTraj_seed = random.randint(0, 100000)
        args.random_motion_seed = random.randint(0, 100000)
        args.max_trans = motion_severity_settings[severity]["max_mot"]
        args.max_rot = motion_severity_settings[severity]["max_mot"]
        args.num_motion_events = motion_severity_settings[severity]["num_event"]
        gt_motion_params = gen_rand_mot_params_interShot(args.Ns, args.max_trans, args.max_rot, args.random_motion_seed, args.num_motion_events, args.Ns)
        traj = sim_motion_get_traj(args, mask3D)
        masked_img3D_coil = motion_correction_NUFFT(kspace3D_masked, torch.zeros(args.Ns, 6).cuda(args.gpu), traj, weight_rot=True, args=args,
                                                        do_dcomp=True, num_iters_dcomp=3)
        masked_corrupted_kspace3D = motion_corruption_NUFFT(kspace3D_masked, masked_img3D_coil, gt_motion_params, traj, weight_rot=True, args=args,
                                                        max_coil_size=2)
        
        masked_corrupted_img3D_coil = ifft2c_ndim(masked_corrupted_kspace3D, 3)
        masked_corrupted_img3D = complex_mul(masked_corrupted_img3D_coil, smaps3D_conj).sum(dim=0, keepdim=False)

        # Unet reconstruction:
        recon = Unet_recon(
            input_img=masked_corrupted_img3D,
            binary_background_mask=binary_background_mask,
            args=args,
        )
        recon = registration(recon, reference_image)
        print("Unet reconstruction shape: ", recon.shape)
        # L1 reconstruction:
        L1_args = SimpleNamespace(
            gpu = args.gpu,
            nufft_norm = args.nufft_norm,
            )
        recon_model = L1minModuleBase(
            smaps3D=smaps3D,
            binary_background_mask=binary_background_mask,
            masked_corrupted_kspace3D=masked_corrupted_kspace3D,
            mask3D=torch.tensor(mask3D).cuda(args.gpu),
            gpu=args.gpu,                             
            L1min_lr=5e7,          
            L1min_lambda=1e-3,
            L1min_num_steps=50,
            traj=traj,
            pred_motion_params=None,  
            L1min_optimizer_type="SGD",
            args=L1_args
        )
        reconstructed_image = recon_model.run_L1min()
        # Save the Unet reconstruction:
        with h5py.File(os.path.join(path, f"severity_{severity}_seed_{args.random_motion_seed}.h5"), 'w') as f:
            f.create_dataset("unet_recon", data=recon)
            f.create_dataset("l1_recon", data=complex_abs(reconstructed_image).squeeze().cpu().numpy())