# PMoC3D
<p align="left">
  <a href="https://huggingface.co/datasets/mli-lab/PMoC3D">
    <img
      src="https://img.shields.io/badge/Hugging%20Face-PMoC3D-FF9900?style=for-the-badge&logo=huggingface&logoColor=origin"
      alt="Hugging Face: PMoC3D"
      height="28"
    />
  </a>
</p>


## Score Calculation:
Step 1: run registeration using ANTs:
```bash
python metric_cal/run_registration_all_recons.py --recon_path /your_path/PMoC3D/reconstruction --base_save_folder /your_save_folder
```

Step 2: Calculate the brain mask for the reference:
need to install fsl:
curl -Ls https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/getfsl.sh | sh -s
apt-get install dc
```bash
python metric_cal/run_brain_mask_extraction.py --recon_path /media/ssd0/PMoC3D/reconstruction --base_save_folder /media/ssd0/PMoC3D/dervatives
```

Step 3: calculate the score:
```bash
python metric_cal/main_score_cal.py --registered_recon_path /media/ssd0/PMoC3D/dervatives/registered_recon --brain_mask_path /media/ssd0/PMoC3D/dervatives/brain_mask --reference_path /media/ssd0/PMoC3D/reconstruction/Reference --score_save_path /kun/PMoC3D/Results --metrics 'dists'
```