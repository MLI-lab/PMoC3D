'''
Running the code to calculate the score evaluation on the PMoC3D reconstructions.
'''
import argparse
import json
import os
from tqdm import tqdm
import torch
import numpy as np

from metrics import metrics
from preprocess.utils import sort_out_zero_slices, normalize_percentile

def main(args):
    baseline_list = ['AltOpt', 'MotionTTT', 'stacked_unet']
    score_list = args.metrics

    for score in score_list:
        score_record = {}
        # for baseline in tqdm(baseline_list, desc=f"Calculating {score}"):
        for baseline in baseline_list:
            score_record[baseline] = {}
            # for sub in range(8):
            for sub in tqdm(range(8), desc=f"Calculating {score} for {baseline}"):
                brain_mask = torch.load(os.path.join(args.brain_mask_path, f"S{sub+1}_mask.pt")).numpy()
                for run_id in range(3):
                    scan_id = f"S{sub+1}_{run_id+1}"
                    ref = torch.load(os.path.join(args.reference_path, f"S{sub+1}_0.pt"))
                    recon = torch.load(os.path.join(args.registered_recon_path, baseline, f"{scan_id}.pt"))

                    recon = normalize_percentile(recon.numpy())
                    ref = normalize_percentile(ref.numpy())

                    recon *= brain_mask
                    ref *= brain_mask

                    recon, ref = sort_out_zero_slices(recon, ref)

                    recon = torch.from_numpy(recon)
                    ref = torch.from_numpy(ref)

                    if score in ['psnr', 'ap']:
                        score_record[baseline][scan_id] = metrics[score](recon, ref)
                    elif score in ['tg', 'aes']:
                        score_record[baseline][scan_id] = metrics[score](recon, score_direction=1)
                    else:
                        score_record[baseline][scan_id] = metrics[score](recon, ref, score_direction=1, gpu=args.gpu)

        os.makedirs(args.score_save_path, exist_ok=True)
        with open(os.path.join(args.score_save_path, f"{score}_scores.json"), 'w') as f:
            json.dump(score_record, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate scores for PMoC3D reconstructions")
    parser.add_argument('--registered_recon_path', type=str, required=True)
    parser.add_argument('--brain_mask_path', type=str, required=True)
    parser.add_argument('--reference_path', type=str, required=True)
    parser.add_argument('--score_save_path', type=str, required=True)
    parser.add_argument('--metrics',type=str,nargs='+',default=['psnr', 'ssim', 'ap', 'tg', 'aes'],
                        help="List of metrics to evaluate (e.g., --metrics psnr ssim)")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU id, -1 for CPU")

    args = parser.parse_args()
    main(args)