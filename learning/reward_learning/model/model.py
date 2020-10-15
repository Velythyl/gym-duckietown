import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np

class CIFARNet(pl.LightningModule):
    def __init__(self):
        super(CIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.to("cuda")
        self.float()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        # using TrainResult to enable logging
        result = pl.TrainResult(loss)
        result.log('loss', loss)

        return result


    # why does step only act once? aka why is epoch end called after a single step??
    def validation_step(self, batch, idx):
        x, y = batch

        loss = F.mse_loss(self(x), y)

        return {"val_loss": loss}

    def validation_epoch_end(self, outs):
        outs = torch.stack([x["val_loss"] for x in outs])
        loss = torch.mean(outs)

        # TODO why doesn't this support the same type of logging as train_step?
        self.logger.experiment.add_scalars("loss", {"val_loss": loss}, self.global_step)

        return {"val_loss": loss, "loss": loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)