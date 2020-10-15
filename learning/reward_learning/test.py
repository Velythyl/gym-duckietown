import torch
import numpy as np

from learning.reward_learning.model.model import CIFARNet

model = CIFARNet.load_from_checkpoint("./lightning_logs/version_0/checkpoints/epoch=9.ckpt")
valid_obs = torch.from_numpy(
        np.transpose(np.load("./dataset/obs.npy")[50000:], (0, 3, 1, 2)).astype(float)).cuda().float()
print(model(valid_obs))