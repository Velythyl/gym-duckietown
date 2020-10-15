import random

import numpy as np
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from learning.reward_learning.dataset.dataset import TrainSet, ValSet
from learning.reward_learning.model.model import CIFARNet

import pytorch_lightning as pl

def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
seed(random.randint(0, 9999999))

train_loader = DataLoader(TrainSet(), batch_size=128,
                        shuffle=True, num_workers=1)
val_loader = DataLoader(ValSet(), batch_size=32, shuffle=False, num_workers=1)

model = CIFARNet()
trainer = pl.Trainer(
    gpus=1,
    early_stop_callback=EarlyStopping(
                monitor='val_loss',
                patience=6,
                strict=True,
                verbose=True,
                mode='min'
            )
)
trainer.fit(model, train_loader, val_loader)