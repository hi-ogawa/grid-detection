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

import matplotlib.pyplot as plt

from .model import make_model


def make_augmentation():
  return VT.Compose([
    VT.RandomApply([VT.GaussianBlur(3)], p=0.5),
    VT.ColorJitter(0.3, 0.3, 0.3, 0.3),
  ])


def make_normalization():
  return VT.Compose([
    VT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])


def make_figure(t, label):
  image = VF.to_pil_image(t.cpu())
  points = label.reshape([4, 2]).cpu().numpy()
  fig, ax = plt.subplots()
  ax.set(xticks=[], yticks=[])
  ax.imshow(image)
  ax.scatter(points[:, 0], points[:, 1], s=100, c=['#fff', '#f44', '#4f4', '#44f'], edgecolors='black', linewidth=2)
  return fig


class Dataset():
  def __init__(self, directory):
    self.pngs = sorted(list(pathlib.Path(directory).glob("*.png")))
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
  def __init__(
      self,
      version=3, dropout=0.2, hidden_size=512, hidden_depth=2,
      lr=0.001, lr_type=1, lr_step=30, lr_gamma=1, lr_patience=5,
      log_image_interval=1, augmentation=True):
    super().__init__()
    self.lr = lr
    self.lr_type = lr_type
    self.lr_step = lr_step
    self.lr_gamma = lr_gamma
    self.lr_patience = lr_patience
    self.log_image_interval = log_image_interval
    self.model = make_model(version, dropout, hidden_size, hidden_depth)
    self.augment = make_augmentation() if augmentation else None
    self.normalize = make_normalization()
    self.train_l1_loss = torchmetrics.MeanAbsoluteError()
    self.train_l2_loss = torchmetrics.MeanSquaredError()
    self.val_l1_loss = torchmetrics.MeanAbsoluteError()
    self.val_l2_loss = torchmetrics.MeanSquaredError()
    print(self.model)

  def forward(self, x):
    return self.model(self.normalize(x))

  def on_train_epoch_start(self):
    self.train_l1_loss.reset()
    self.train_l2_loss.reset()
    lr = self.optimizers().optimizer.param_groups[0]['lr']
    self.log('lr', lr)
    print('[lr]', lr)

  def training_step(self, batch, batch_idx):
    x, y = batch
    if self.augment:
      x = self.augment(x)
    y_hat = self.forward(x)
    l1_loss = self.train_l1_loss(y, y_hat)
    l2_loss = self.train_l2_loss(y, y_hat)
    self.log('train_l1_loss', l1_loss)
    self.log('train_l2_loss', l2_loss)
    return l2_loss

  def training_epoch_end(self, _):
    l1_loss = self.train_l1_loss.compute()
    l2_loss = self.train_l2_loss.compute()
    self.log('train_l1_loss--epoch', l1_loss)
    self.log('train_l2_loss--epoch', l2_loss)
    print('[train_loss]', float(l1_loss), float(l2_loss))

  def on_validation_epoch_start(self):
    self.val_l1_loss.reset()
    self.val_l2_loss.reset()

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.forward(x)
    l1_loss = self.val_l1_loss(y, y_hat)
    l2_loss = self.val_l2_loss(y, y_hat)
    self.log('val_l1_loss', l1_loss)
    self.log('val_l2_loss', l2_loss)
    if batch_idx % self.log_image_interval == 0:
      fig = make_figure(x[0], y_hat[0])
      self.logger.experiment.add_figure(f"val_image_{batch_idx}", fig, global_step=self.trainer.global_step)

  def validation_epoch_end(self, _):
    l1_loss = self.val_l1_loss.compute()
    l2_loss = self.val_l2_loss.compute()
    self.log('val_l1_loss--epoch', l1_loss)
    self.log('val_l2_loss--epoch', l2_loss)
    print('[val_loss]', float(l1_loss), float(l2_loss))

  def configure_optimizers(self):
    opt = torch.optim.Adam(self.parameters(), lr=self.lr)
    sch1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, patience=self.lr_patience)
    sch2 = torch.optim.lr_scheduler.StepLR(opt, verbose=True, step_size=self.lr_step, gamma=self.lr_gamma)
    sch = [0, sch1, sch2][self.lr_type]
    return {
      "optimizer": opt,
      "lr_scheduler": {
        "scheduler": sch,
        "monitor": "val_l2_loss--epoch"
      }
    }


if __name__ == '__main__':
  cli = pl.utilities.cli.LightningCLI(Model, DataModule)
