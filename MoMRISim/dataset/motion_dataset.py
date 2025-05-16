import os
import random
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from itertools import combinations
import torch
from MoMRISim.util.utils import get_preprocess_fn
from torchvision import transforms



class TwoAFCDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train",load_size: int = 224,
                 interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
                 preprocess: str = "DEFAULT", **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.preprocess_fn = get_preprocess_fn(preprocess, load_size, interpolation)
        # Get the name of all directories in the root directory
        self.subdirs = []
        for d in os.listdir(root_dir):
            full_path = os.path.join(root_dir, d)
            if os.path.isdir(full_path):
                self.subdirs.append(d)
        self.subdirs.sort()
        # List all combination
        self.samples = []
        for subdir_name in self.subdirs:
            subdir_path = os.path.join(self.root_dir, subdir_name)
            # Check if the folder contains ref_img3D.h5
            ref_file_path = os.path.join(subdir_path, "ref_img3D.h5")
            if not os.path.exists(ref_file_path):
                continue
            # Collect all motion artifact affected file:
            severity_files = []
            for filename in os.listdir(subdir_path):
                if filename.startswith("severity_") and filename.endswith(".h5"):
                    severity_files.append(filename)
            severity_files.sort()
            # If there are less than 2 severity files, skip this folder
            if len(severity_files) < 2:
                continue
            # Get all pairs
            severity_pairs = list(combinations(severity_files, 2))
            for (a_file, b_file) in severity_pairs:
                sevA = self._parse_severity(a_file)
                sevB = self._parse_severity(b_file)
                # Skip if the severity are the same for 2 files
                if sevA == sevB:
                    continue
                self.samples.append((subdir_name, a_file, b_file))
        print(f"Total volume pairs: {len(self.samples)}")

    def __len__(self):
        # return len(self.samples)
        if self.split == "train":
            return len(self.samples)#*10
        elif self.split == "val":
            return 100

    def __getitem__(self, idx):
        sample_idx = random.randint(0, len(self.samples)-1)
        subdir_name, severityA_file, severityB_file = self.samples[sample_idx]
        subdir_path = os.path.join(self.root_dir, subdir_name)
        # =============== Read the reference 3D volume ===============
        ref_file_path = os.path.join(subdir_path, "ref_img3D.h5")
        with h5py.File(ref_file_path, 'r') as f_ref:
            ref_volume_3d = f_ref["reference"][:]  # shape: (D,H,W)
        # =============== Read the severity of the paired data ===============
        severityA = self._parse_severity(severityA_file)  # int
        severityB = self._parse_severity(severityB_file)  # int
        severityA_path = os.path.join(subdir_path, severityA_file)
        severityB_path = os.path.join(subdir_path, severityB_file)
        dataset_A = random.choice(["unet_recon", "l1_recon"])
        dataset_B = random.choice(["unet_recon", "l1_recon"])

        # =============== load severity_A and severity_B files ===============
        with h5py.File(severityA_path, 'r') as fA:
            volA_3d = fA[dataset_A][:]
        with h5py.File(severityB_path, 'r') as fB:
            volB_3d = fB[dataset_B][:]
        # =============== Normalize the 3D volumes ===============
        ref_volume_3d = self.normalize_percentile(ref_volume_3d)
        volA_3d = self.normalize_percentile(volA_3d)
        volB_3d = self.normalize_percentile(volB_3d)
        # =============== Randomly choosing orientation & slice_index ===============
        orientation = random.randint(0, 2)
        # =============== Remove the zero slices ===============
        non_zero_ratio = 0.25
        if orientation == 0:
            slice_pixels = ref_volume_3d.shape[1] * ref_volume_3d.shape[2]
            zero_slices = np.where(np.sum(ref_volume_3d > 0, axis=(1, 2)) / slice_pixels < non_zero_ratio)[0]
        elif orientation == 1:
            slice_pixels = ref_volume_3d.shape[0] * ref_volume_3d.shape[2]   
            zero_slices = np.where(np.sum(ref_volume_3d > 0, axis=(0, 2)) / slice_pixels < non_zero_ratio)[0]
        else:
            slice_pixels = ref_volume_3d.shape[0] * ref_volume_3d.shape[1] 
            zero_slices = np.where(np.sum(ref_volume_3d > 0, axis=(0, 1)) / slice_pixels < non_zero_ratio)[0]
        # print(ref_volume_3d.shape)
        ref_volume_3d = np.delete(ref_volume_3d, zero_slices, axis=orientation)
        volA_3d = np.delete(volA_3d, zero_slices, axis=orientation)
        volB_3d = np.delete(volB_3d, zero_slices, axis=orientation)
        # print(ref_volume_3d.shape)
        # Get the slice index
        slice_index =slice_index = random.randint(0, ref_volume_3d.shape[orientation] - 1)

        ref_slice = self._get_slice(ref_volume_3d, orientation, slice_index)
        motionA_slice = self._get_slice(volA_3d, orientation, slice_index)
        motionB_slice = self._get_slice(volB_3d, orientation, slice_index)

        # p=1 => severityA > severityB
        # p=0 => severityA < severityB
        # if severityA > severityB:
        #     p = 1.0
        # else:
        #     p = 0.0
        p = 1.0 if severityA > severityB else 0.0
        target = torch.tensor(p, dtype=torch.float32)
        id_str = f"{subdir_name}_A{severityA}_B{severityB}"
        return ref_slice, motionA_slice, motionB_slice, target, id_str

    def _get_slice(self, volume_3d, orientation, slice_index):
        if orientation == 0:
            slice_2d = volume_3d[slice_index, :, :]
        elif orientation == 1:
            slice_2d = volume_3d[:, slice_index, :]
        else:
            slice_2d = volume_3d[:, :, slice_index]
        # slice_2d = self.normalize_percentile(slice_2d)
        tensor_2d = torch.from_numpy(slice_2d).unsqueeze(0).repeat(3,1,1)
        if self.preprocess_fn is not None:
            # np_img = (tensor_2d.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # pil_img = Image.fromarray(np_img)
            pil_img = transforms.ToPILImage()(tensor_2d)
            processed_img = self.preprocess_fn(pil_img)  # 预处理函数通常返回 [3,224,224] 的 Tensor
            return processed_img
        else:
            return tensor_2d

    def _parse_severity(self, filename: str) -> int:
        """
        Get severity=5 from filename like "severity_5_seed_12345.h5" 
        My filename format: severity_{INT}_seed_{xxxx}.h5
        """
        parts = filename.split('_')
        severity_val = int(parts[1])
        return severity_val
    @staticmethod
    def normalize_percentile(img, lower_percentile=1, upper_percentile=99.9, clip=True):
        """ Normalization to the lower and upper percentiles """
        # img = img.astype(np.float32)
        lower = np.percentile(img, lower_percentile)
        upper = np.percentile(img, upper_percentile)
        img = (img - lower) / (upper - lower+1e-20)
        if clip:
            img = np.clip(img, 0, 1)
        return img