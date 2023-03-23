from torch.utils.data import Dataset

import os
import numpy as np

import sys
import os
import pathlib
from pathlib import Path

import pickle
from absl import app
from absl import flags
import h5py

import torch

class trajDataset(Dataset):

    def __init__(self, path, split, transform=None):
        self.path = os.path.join(path, split + ".h5")
        self.transform = transform
        self.data = h5py.File(self.path, 'r')


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        traj_np = {}
        traj = self.data[str(idx)]
        for key, value in traj.items():
            traj_np[key] = np.array(value)
        return self.transform(traj_np)


