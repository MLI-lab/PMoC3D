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

This repository contains code for downloading the PMoC3D dataset and evaluating 3D reconstruction quality.

## Requirements
We recommend using a Conda environment for managing dependencies.

Step 1: Create and Activate Environment

```bash
conda create -n pmoc3d_env python=3.10.14 pip=24.2
conda activate pmoc3d_env
```

Step 2: Install PyTorch with CUDA Support

Install a CUDA-compatible version of PyTorch suitable for your system.

Step 3: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```
## Download the Dataset

To download the PMoC3D dataset, follow these steps:

1. **Request Access**  
   Visit the [PMoC3D HuggingFace](https://huggingface.co/datasets/mli-lab/PMoC3D) and agree to the Data Usage Agreement (DUA).

2. **Generate a Token**  
   After access is granted, create a Hugging Face access token from your Hugging Face account settings.

3. **Run the Download Script**  
   Use the following command to download the dataset:

   ```bash
   python dataset/download_dataset.py --local_save_path <your_local_save_path> --hugging_face_token <your_hf_token> --mode all
   ```

### Available Modes:
- `all`: Download the full dataset (reconstruction and k-space data).
- `reconstruction`: Download only the 3D reconstruction data.
- `sourcedata`: Download only the raw k-space data.

Replace `<your_local_save_path>` and `<your_hf_token>` with your actual values.

## Score Calculation

We provide reconstruction outputs for three baseline models — MotionTTT, AltOpt, and Stacked UNet — located in the `PMoC3D/reconstruction` directory.

You can calculate reconstruction scores using the following steps:

### Step 1: Run Registration with ANTs

This step registers all reconstructed volumes to the reference using ANTs. Run:

```bash
python metric_cal/run_registration_all_recons.py --recon_path /your_path/PMoC3D/reconstruction --base_save_folder /your_path/PMoC3D/derivatives
```

### Step 2: Extract Brain Mask

First, install FSL (required for brain mask extraction):

```bash
curl -Ls https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/getfsl.sh | sh -s
sudo apt-get install dc
```

Then run:

```bash
python metric_cal/run_brain_mask_extraction.py --recon_path /your_path/PMoC3D/reconstruction --base_save_folder /your_path/PMoC3D/derivatives
```

### Step 3: Calculate Reconstruction Scores

Run the scoring script using the registered reconstructions, extracted brain mask, and the reference data:

```bash
python metric_cal/main_score_cal.py \
  --registered_recon_path /your_path/PMoC3D/derivatives/registered_recon \
  --brain_mask_path /your_path/PMoC3D/derivatives/brain_mask \
  --reference_path /your_path/PMoC3D/reconstruction/Reference \
  --score_save_path /your_path/PMoC3D/results \
  --metrics psnr ssim ap
```

Replace `/your_path` with your actual working directory path.

