import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as VT
import torchvision.transforms.functional as VF

import torchmetrics

import pytorch_lightning as pl
import pytorch_lightning.utilities.cli

import PIL.Image

import pathlib
from functools import partial


def random_noise(t, mean=0, std=1):
  return t + mean + torch.randn(t.size()) * std


def make_augumentation():
  return VT.Compose([
    VT.RandomApply([VT.GaussianBlur(5)], p=0.5),
    VT.RandomApply([partial(random_noise, std=0.1)], p=0.5),
    VT.ColorJitter(0.4, 0.4, 0.4, 0.4),
  ])


class Dataset():
  def __init__(self, directory):
    self.pngs = list(pathlib.Path(directory).glob("*.png"))
    self.labels = torch.load(f"{directory}/labels.pt")

  def __getitem__(self, i):
    png = self.pngs[i]
    label = self.labels[i]
    image = PIL.Image.open(png)
    x = VF.to_tensor(image)
    y = torch.FloatTensor(label).reshape(8)
    return x, y

  def __len__(self):
    return len(self.pngs)


class DataModule(pl.LightningDataModule):
  def __init__(self, directory: str, batch_size=8, val_fraction=0.05, num_workers=0):
    super().__init__()
    self.dataset = Dataset(directory)
    self.loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
    len_total = len(self.dataset)
    len_val = int(val_fraction * len_total)
    len_train = len_total - len_val
    self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [len_train, len_val])

  def train_dataloader(self):
      return torch.utils.data.DataLoader(self.train_dataset, **self.loader_kwargs)

  def val_dataloader(self):
      return torch.utils.data.DataLoader(self.val_dataset, **self.loader_kwargs)


class Model(pl.LightningModule):
  def __init__(self, dropout=0.2):
    super().__init__()
    self.sequential = nn.Sequential(
      # --
      nn.Conv2d(3, 32, kernel_size=5),
      nn.ReLU(),
      nn.MaxPool2d(2),
      # --
      nn.Conv2d(32, 64, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool2d(2),
      # --
      nn.Conv2d(64, 128, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool2d(2),
      # --
      nn.Conv2d(128, 128, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool2d(2),
      # --
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),
      nn.Dropout(dropout),
      # --
      nn.Linear(128, 8)
    )
    self.augument = make_augumentation()

  def forward(self, x):
    return self.sequential(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    x = self.augument(x)
    y_hat = self.forward(x)
    loss = F.mse_loss(y, y_hat)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.forward(x)
    loss = F.mse_loss(y, y_hat)
    self.log('val_loss', loss)

  def configure_optimizers(self):
    opt = torch.optim.Adam(self.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=1)
    return dict(optimizer=opt, scheduler=sch)


if __name__ == '__main__':
  cli = pl.utilities.cli.LightningCLI(Model, DataModule)
