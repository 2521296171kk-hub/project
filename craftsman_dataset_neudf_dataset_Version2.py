import os
import numpy as np
from torch.utils.data import Dataset
import torch

class NeUDFGridDataset(Dataset):
    def __init__(self, data_dir, resolution=64, max_samples=None):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        if max_samples is not None:
            self.files = self.files[:max_samples]
        self.resolution = resolution

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        # normalize to [0,1]
        v = arr.astype(np.float32)
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        v = torch.from_numpy(v).unsqueeze(0)  # (1, D, H, W)
        return v