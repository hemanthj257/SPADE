import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SAROpticalDataset(Dataset):
    def __init__(self, sar_dir, optical_dir, img_size=(256, 256)):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.img_size = img_size
        self.sar_images = sorted(os.listdir(sar_dir))
        self.optical_images = sorted(os.listdir(optical_dir))

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        sar_img_path = os.path.join(self.sar_dir, self.sar_images[idx])
        optical_img_path = os.path.join(self.optical_dir, self.optical_images[idx])

        # Load SAR image in grayscale and normalize to [0, 1]
        sar_img = cv2.imread(sar_img_path, cv2.IMREAD_GRAYSCALE)
        if sar_img is None:
            raise FileNotFoundError(f"Failed to load SAR image at path: {sar_img_path}")
        sar_img = cv2.resize(sar_img, self.img_size) / 255.0
        sar_img = np.expand_dims(sar_img, axis=0)  # Shape: (1, H, W)

        # Load optical image, convert from BGR to RGB, and normalize to [-1, 1]
        optical_img = cv2.imread(optical_img_path)
        if optical_img is None:
            raise FileNotFoundError(f"Failed to load optical image at path: {optical_img_path}")
        optical_img = cv2.cvtColor(optical_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        optical_img = cv2.resize(optical_img, self.img_size)
        optical_img = (optical_img / 127.5) - 1  # Normalize to [-1, 1]
        optical_img = np.transpose(optical_img, (2, 0, 1))  # Shape: (3, H, W)

        return torch.from_numpy(sar_img).float(), torch.from_numpy(optical_img).float()