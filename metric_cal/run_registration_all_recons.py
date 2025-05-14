'''
Code for registration of all reconstructions
'''
import os
import argparse
import torch
from tqdm import tqdm
from preprocess.registration import align_volumes_with_ants

def main(recon_path: str, base_save_folder: str):
    save_path = os.path.join(base_save_folder, "registered_recon")
    os.makedirs(save_path, exist_ok=True)

    baseline_name = ['AltOpt','L1_noMoCo','MotionTTT','stacked_unet']
    scan_index = [
        'S1_1','S1_2','S1_3',
        'S2_1','S2_2','S2_3',
        'S3_1','S3_2','S3_3',
        'S4_1','S4_2','S4_3',
        'S5_1','S5_2','S5_3',
        'S6_1','S6_2','S6_3',
        'S7_1','S7_2','S7_3',
        'S8_1','S8_2','S8_3',
    ]

    for method in baseline_name:
        method_save_path = os.path.join(save_path, method)
        os.makedirs(method_save_path, exist_ok=True)

        scan_index_tqdm = tqdm(scan_index, desc=f"Processing {method}", postfix={"method": method})

        for si in scan_index_tqdm:
            subject_ind = si.split('_')[0]
            recon_file = os.path.join(recon_path, method, f"{si}.pt")
            target_file = os.path.join(recon_path, "Reference", f"{subject_ind}_0.pt")
            save_file_name = os.path.join(method_save_path, f"{si}.pt")

            if os.path.exists(recon_file) and os.path.exists(target_file):
                recon = torch.load(recon_file, map_location=torch.device('cpu')).numpy()
                target = torch.load(target_file, map_location=torch.device('cpu')).numpy()
                aligned_recon = align_volumes_with_ants(recon, target)
                torch.save(torch.tensor(aligned_recon), save_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register all reconstruction volumes")
    parser.add_argument("--recon_path", type=str, required=True, help="Path to reconstruction volumes")
    parser.add_argument("--base_save_folder", type=str, required=True, help="Base folder to save registered results")
    args = parser.parse_args()

    main(args.recon_path, args.base_save_folder)