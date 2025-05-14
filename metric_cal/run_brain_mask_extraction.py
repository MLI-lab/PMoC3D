'''
Code for registration of all reconstructions
'''
import os
import argparse
import torch
from tqdm import tqdm
import nibabel as nib
import numpy as np
import os
import subprocess


def main(recon_path: str, base_save_folder: str):
    os.environ['FSLDIR'] = '/root/fsl'
    os.environ['PATH'] = f"{os.environ['FSLDIR']}/bin:" + os.environ['PATH']
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI'

    nii_save_path = os.path.join(base_save_folder, "Reference_nii")
    os.makedirs(nii_save_path, exist_ok=True)
    brain_nii_folder = os.path.join(base_save_folder, "brain_mask_nii")
    os.makedirs(brain_nii_folder, exist_ok=True)
    brain_mask_folder = os.path.join(base_save_folder, "brain_mask")
    os.makedirs(brain_mask_folder, exist_ok=True)
    for i in range(8):
        # Step 1: save all reference reconstruction into .nii files:
        scan_index = f"S{i+1}"
        target_volume = torch.load(os.path.join(recon_path, "Reference",scan_index + "_0.pt"),map_location=torch.device('cpu')).numpy()
        output_nii_file = os.path.join(nii_save_path, scan_index+'.nii')
        affine = np.eye(4)
        nii_image = nib.Nifti1Image(target_volume, affine)
        nib.save(nii_image, output_nii_file)
        print(f"NIfTI file successfully saved to: {output_nii_file}")
    
        # Step 2: Brain extraction using FSL BET:
        output_brain_nii_file = os.path.join(brain_nii_folder, scan_index)
        bet_command = [
            'bet', output_nii_file, output_brain_nii_file,
            '-R', '-f', '0.9', '-g', '0.35', '-m'
        ]

        try:
            result = subprocess.run(
                bet_command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("BET completed successfully.")

        except subprocess.CalledProcessError as e:
            print("Error during BET execution:")
            print(e.stderr.decode('utf-8'))
        
        brain_mask_file = f"{output_brain_nii_file}_mask.nii"
        mask_img = nib.load(brain_mask_file)
        mask_data = mask_img.get_fdata()
        # Convert NumPy array to a PyTorch Tensor
        mask_tensor = torch.tensor(mask_data, dtype=torch.float32)
        # Save the Tensor as a .pt file
        pt_file_path = os.path.join(brain_mask_folder,f"{scan_index}_mask.pt")  # Replace with the desired output .pt file path
        torch.save(mask_tensor, pt_file_path)
        print(f"Saved PyTorch tensor to: {pt_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register all reconstruction volumes")
    parser.add_argument("--recon_path", type=str, required=True, help="Path to reconstruction volumes")
    parser.add_argument("--base_save_folder", type=str, required=True, help="Base folder to save registered results")
    args = parser.parse_args()

    main(args.recon_path, args.base_save_folder)