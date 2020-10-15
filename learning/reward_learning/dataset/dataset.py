import torch
from torch.utils.data import Dataset
import numpy as np
import pathlib


class ValSet(Dataset):
    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        path = str(pathlib.Path(__file__).parent.absolute())
        self.obs = np.load(path + "/obs.npy")[50000:]
        self.rew = np.load(path + "/rew.npy")[50000:]

    def __len__(self):
        return self.rew.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            idx = idx[0]

        obs = np.transpose(self.obs[idx], (2, 0, 1)).astype(float)
        obs = torch.from_numpy(obs)
        obs = obs.float()

        return obs, torch.Tensor([self.rew[idx]]).float()


class TrainSet(Dataset):

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        path = str(pathlib.Path(__file__).parent.absolute())
        self.obs = np.load(path + "/obs.npy")[:50000]
        self.rew = np.load(path + "/rew.npy")[:50000]

    def __len__(self):
        return self.rew.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            idx = idx[0]

        obs = np.transpose(self.obs[idx], (2, 0, 1)).astype(float)
        obs = torch.from_numpy(obs)
        obs = obs.float()

        return obs, torch.Tensor([self.rew[idx]]).float()
